import abc
import Tensor
import networks
import typing
import strategies.conversion as cv
import utilities
from maraboupy import Marabou
from maraboupy import MarabouUtils
from maraboupy import MarabouCore
import onnx
import os
import numpy as np
import eran
import constraints
import ai_milp
import nodes
import scipy.io as sio
import subprocess


class Property(abc.ABC):
    """
    An abstract class used to represent a generic property for a NeuralNetwork.
    """


class SMTLIBProperty(Property):
    """
    A concrete class used to represent a generic property for a NeuralNetwork expressed as a SMTLIB query.

    Attributes
    ----------
    smtlib_path : str
        Filepath for the SMTLIB file in which the property is defined.

    """

    def __init__(self, smtlib_path: str):
        self.smtlib_path = smtlib_path


class LocalRobustnessProperty(Property):
    """
    A concrete class used to represent a local robustness property for a NeuralNetwork.
    Formally the property check if the counterexample (i.e., the adversarial example) exists, therefore
    when the verification strategy check such property it should return True if the adversarial example exist and
    false otherwise.

    Attributes
    ----------
    data : Tensor
        Original data used to determine the local robustness.
    target : int
        If targeted is True then it is the desired target for the adversarial, otherwise it is the correct target of
        data.
    targeted : bool
        Flag which is True if the robustness property is targeted, False otherwise.
    norm : str
        Norm type used to determine the local robustness. At present the only acceptable value is Linf.
    epsilon : float
        Magnitude of the acceptable perturbation.
    bounds: list
        List of (lower_bound, upper_bound) for the data.

    """

    def __init__(self, data: Tensor.Tensor, target: int, targeted: bool, norm: str, epsilon: float, bounds: list):

        self.data = data
        self.target = target
        self.targeted = targeted
        if norm != "Linf":
            raise NotImplementedError
        self.norm = norm
        self.epsilon = epsilon
        self.bounds = bounds


class VerificationStrategy(abc.ABC):
    """
    An abstract class used to represent a Verification Strategy.

    Methods
    ----------
    verify(NeuralNetwork, Property)
        Verify that the neural network of interest satisfy the property given as argument
        using a verification strategy determined in the concrete children.

    """

    @abc.abstractmethod
    def verify(self, network: networks.NeuralNetwork, prop: Property) -> (bool, typing.Optional[Tensor.Tensor]):
        """
        Verify that the neural network of interest satisfy the property given as argument
        using a verification strategy determined in the concrete children.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to train.
        prop : Dataset
            The property which the neural network must satisfy.

        Returns
        ----------
        bool
            True is the neural network satisfy the property, False otherwise.

        """
        pass


class MarabouVerification(VerificationStrategy):
    """
    Class used to represent the verification strategy based on the Marabou verification tool.
    The code of the tool can be found at https://github.com/NeuralNetworkVerification/Marabou.
    The tool paper can be found at http://aisafety.stanford.edu/marabou/MarabouCAV2019.pdf.

    Methods
    ----------
    verify(NeuralNetwork, Property)
        Verify that the neural network of interest satisfy the property given as argument
        using the Marabou verification tool.

    """

    def verify(self, network: networks.NeuralNetwork, prop: Property) -> (bool, typing.Optional[Tensor.Tensor]):
        """
        Verify that the neural network of interest satisfy the property given as argument
        using the Marabou verification tool.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to train.
        prop : Dataset
            The property which the neural network must satisfy.

        Returns
        ----------
        (bool, Optional[Tensor])
            True and None if the neural network satisfy the property, False and the counterexample otherwise.

        """
        if isinstance(prop, SMTLIBProperty):
            targeted, bounds, target = utilities.parse_linf_robustness_smtlib(prop.smtlib_path)
        elif isinstance(prop, LocalRobustnessProperty):
            targeted = prop.targeted
            target = prop.target
            bounds = []
            for i in range(len(prop.data)):

                if prop.data[i] + prop.epsilon > prop.bounds[i][1]:
                    ub = prop.bounds[i][1]
                else:
                    ub = prop.data[i] + prop.epsilon

                if prop.data[i] - prop.epsilon < prop.bounds[i][0]:
                    lb = prop.bounds[i][0]
                else:
                    lb = prop.data[i] - prop.epsilon

                bounds.append((lb, ub))
        else:
            raise NotImplementedError

        if not targeted:
            raise NotImplementedError

        onnx_rep = cv.ONNXConverter().from_neural_network(network)
        onnx.save_model(onnx_rep.onnx_network, "temp/onnx_network.onnx")

        marabou_onnx_net = Marabou.read_onnx("temp/onnx_network.onnx")
        os.remove("temp/onnx_network.onnx")
        input_vars = marabou_onnx_net.inputVars[0][0]
        output_vars = marabou_onnx_net.outputVars

        assert(len(bounds) == len(input_vars))

        for i in range(len(input_vars)):
            marabou_onnx_net.setLowerBound(input_vars[i], bounds[i][0])
            marabou_onnx_net.setUpperBound(input_vars[i], bounds[i][1])

        for i in range(len(output_vars)):
            if i != target:
                MarabouUtils.addInequality(marabou_onnx_net, [output_vars[i], output_vars[target]], [1, -1], 0)

        options = MarabouCore.Options()
        # options._verbosity = 2

        vals, stats = marabou_onnx_net.solve(options=options)

        counterexample = None
        if not vals:
            sat = False
        else:
            sat = True
            counterexample = [val for val in vals.values()]
            counterexample = np.array(counterexample)

        return sat, counterexample


class ERANVerification(VerificationStrategy):
    """
    Class used to represent the verification strategy based on the ERAN verification tool.
    The code of the tool can be found at https://github.com/eth-sri/eran.
    The tool paper can be found at https://files.sri.inf.ethz.ch/website/papers/RefineZono.pdf.

    Attributes
    ----------
    complete : bool
        Flag specifying where to use complete verification or not.

    Methods
    ----------
    verify(NeuralNetwork, Property)
        Verify that the neural network of interest satisfy the property given as argument
        using the ERAN verification tool.

    """

    def __init__(self, complete: bool = True):
        self.complete = complete
        self.timeout_lp = 1
        self.timeout_milp = 1
        self.use_default_heuristic = True
        self.domain = "deepzono"

    def verify(self, network: networks.NeuralNetwork, prop: Property) -> (bool, typing.Optional[Tensor.Tensor]):
        """
        Verify that the neural network of interest satisfy the property given as argument
        using the ERAN verification tool. At present the only domain we consider is deepzono.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to train.
        prop : Dataset
            The property which the neural network must satisfy.

        Returns
        ----------
        (bool, Optional[Tensor])
            True and None if the neural network satisfy the property, False and the counterexample otherwise.

        """

        if not isinstance(prop, LocalRobustnessProperty):
            raise NotImplementedError

        if not isinstance(network, networks.SequentialNetwork):
            raise NotImplementedError

        if prop.targeted:
            raise NotImplementedError

        lb_image = []
        ub_image = []
        for i in range(len(prop.data)):

            if prop.data[i] + prop.epsilon > prop.bounds[i][1]:
                ub_image.append(prop.bounds[i][1])
            else:
                ub_image.append(prop.data[i] + prop.epsilon)

            if prop.data[i] - prop.epsilon < prop.bounds[i][0]:
                lb_image.append(prop.bounds[i][0])
            else:
                lb_image.append(prop.data[i] - prop.epsilon)

        lb_image = np.array(lb_image)
        ub_image = np.array(ub_image)
        orig_image = prop.data

        onnx_model_path = "temp/onnx_network.onnx"
        onnx_rep = cv.ONNXConverter().from_neural_network(network)
        onnx.save_model(onnx_rep.onnx_network, onnx_model_path)

        onnx_model = onnx.load(onnx_model_path)
        # onnx.checker.check_model(onnx_model)

        eran_instance = eran.ERAN(onnx_model, is_onnx=True)

        spec_lb = np.copy(orig_image)
        spec_ub = np.copy(orig_image)
        label, nn, nlb, nub = eran_instance.analyze_box(spec_lb, spec_ub, self.domain, self.timeout_lp,
                                                        self.timeout_milp, self.use_default_heuristic)

        if label == prop.target:

            spec_lb = np.copy(lb_image)
            spec_ub = np.copy(ub_image)

            perturbed_label, _, nlb, nub = eran_instance.analyze_box(spec_lb, spec_ub, self.domain, self.timeout_lp,
                                                                     self.timeout_milp, self.use_default_heuristic)

            if perturbed_label == label:
                return False, None
            elif self.complete:
                output_constraints = constraints.get_constraints_for_dominant_label(label, 10)
                verified_flag, adv_image = ai_milp.verify_network_with_milp(nn, spec_lb, spec_ub, nlb, nub,
                                                                            output_constraints)
                if verified_flag:
                    return False, None
                else:
                    return True, np.array(adv_image)
            else:
                return True, None

        return False, None


class MIPVerifyVerification(VerificationStrategy):
    """
    Class used to represent the verification strategy based on the MIPVerify verification tool.
    The code of the tool can be found at https://github.com/vtjeng/MIPVerify.jl.
    The tool paper can be found at https://arxiv.org/abs/1711.07356.

    Attributes
    ----------
    path : str
        Path to the temporary folder used to store the files needed for verification method.

    Methods
    ----------
    verify(NeuralNetwork, Property)
        Verify that the neural network of interest satisfy the property given as argument
        using the MIPVerify verification tool.

    """

    def __init__(self, path: str = "temp/"):
        self.path = path

    def verify(self, network: networks.NeuralNetwork, prop: Property) -> (bool, typing.Optional[Tensor.Tensor]):
        """
        Verify that the neural network of interest satisfy the property given as argument
        using the MIPVerify verification tool.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to train.
        prop : Dataset
            The property which the neural network must satisfy.

        Returns
        ----------
        (bool, Optional[Tensor])
            True and None if the neural network satisfy the property, False and the counterexample otherwise.

        """
        if not isinstance(network, networks.SequentialNetwork):
            raise NotImplementedError

        if not isinstance(prop, LocalRobustnessProperty):
            raise NotImplementedError

        with open(self.path + "mip_img.txt", "w+") as f:
            for pix in prop.data:
                f.write("{}\n".format(pix))
            f.write("{}".format(prop.target))

        self.generate_mipverify_net(network)

        net_name = self.path + network.identifier
        img_name = self.path + "mip_img.txt"
        p = subprocess.run(
            ['julia', os.path.abspath('mipverify_script.jl'), net_name, img_name,
             str(prop.epsilon), str(prop.targeted)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )

        if p.returncode == 0:
            return True, None
        else:
            return False, None

    def generate_mipverify_net(self, network: networks.SequentialNetwork):
        """
        Function to generate the file needed for the passage of the neural network model to
        the julia script.

        Parameters
        ----------
        network : NeuralNetwork
            Network of interest.
        """

        parameters = dict()
        cfg = dict()
        path = self.path + network.identifier
        with open(path + "_cfg.txt", "w+") as f:

            fc_idx = 0
            for node in network.nodes.values():

                if isinstance(node, nodes.FullyConnectedNode):

                    fc_idx += 1
                    parameters["fc{}/weight".format(fc_idx)] = node.weight.T
                    parameters["fc{}/bias".format(fc_idx)] = node.bias

                    cfg["fc{}/in".format(fc_idx)] = node.in_features
                    cfg["fc{}/out".format(fc_idx)] = node.out_features

                    f.write("fc{}\n".format(fc_idx))

                elif isinstance(node, nodes.ReLUNode):
                    f.write("relu\n")

        sio.savemat("{}.mat".format(path), parameters)
        sio.savemat("{}_cfg.mat".format(path), cfg)
