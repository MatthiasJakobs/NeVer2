import strategies.conversion as cv
import nodes
import networks
import datasets
import torch
import math
import torch.nn.functional as funct
import Tensor


def combine_batchnorm1d(linear: nodes.FullyConnectedNode, batchnorm: nodes.BatchNorm1DNode) -> nodes.FullyConnectedNode:
    """
    Utility function to combine a BatchNorm1DNode node with a FullyConnectedNode in a corresponding FullyConnectedNode.

    Parameters
    ----------
    linear : FullyConnectedNode
        FullyConnectedNode to combine.
    batchnorm : BatchNorm1DNode
        BatchNorm1DNode to combine.

    Return
    ----------
    FullyConnectedNode
        The FullyConnectedNode resulting from the fusion of the two input nodes.

    """

    l_weight = torch.from_numpy(linear.weight)
    l_bias = torch.from_numpy(linear.bias)
    bn_running_mean = torch.from_numpy(batchnorm.running_mean)
    bn_running_var = torch.from_numpy(batchnorm.running_var)
    bn_weight = torch.from_numpy(batchnorm.weight)
    bn_bias = torch.from_numpy(batchnorm.bias)
    bn_eps = batchnorm.eps

    fused_bias = torch.div(bn_weight, torch.sqrt(bn_running_var + bn_eps))
    fused_bias = torch.mul(fused_bias, torch.sub(l_bias, bn_running_mean))
    fused_bias = torch.add(fused_bias, bn_bias)

    fused_weight = torch.diag(torch.div(bn_weight, torch.sqrt(bn_running_var + bn_eps)))
    fused_weight = torch.matmul(fused_weight, l_weight)

    fused_linear = nodes.FullyConnectedNode(linear.identifier, linear.in_features, linear.out_features, fused_weight.numpy(),
                                            fused_bias.numpy())

    return fused_linear


def combine_batchnorm1d_net(network: networks.SequentialNetwork) -> networks.SequentialNetwork:
    """
    Utilities function to combine all the FullyConnectedNodes followed by BatchNorm1DNodes in corresponding
    FullyConnectedNodes.

    Parameters
    ----------
    network : SequentialNetwork
        Sequential Network of interest of which we want to combine the nodes.

    Return
    ----------
    SequentialNetwork
        Corresponding Sequential Network with the combined nodes.

    """

    if not network.up_to_date:

        for alt_rep in network.alt_rep_cache:

            if alt_rep.up_to_date:

                if isinstance(alt_rep, cv.PyTorchNetwork):
                    pytorch_cv = cv.PyTorchConverter()
                    network = pytorch_cv.to_neural_network(alt_rep)
                elif isinstance(alt_rep, cv.ONNXNetwork):
                    onnx_cv = cv.ONNXConverter
                    network = onnx_cv.to_neural_network(alt_rep)
                else:
                    raise NotImplementedError
                break

    combined_network = networks.SequentialNetwork(network.identifier + '_combined')

    current_node = network.get_first_node()
    node_index = 1
    while network.get_next_node(current_node) is not None and current_node is not None:

        next_node = network.get_next_node(current_node)
        if isinstance(current_node, nodes.FullyConnectedNode) and isinstance(next_node, nodes.BatchNorm1DNode):
            combined_node = combine_batchnorm1d(current_node, next_node)
            combined_node.identifier = f"Combined_Linear_{node_index}"
            combined_network.add_node(combined_node)
            next_node = network.get_next_node(next_node)

        elif isinstance(current_node, nodes.FullyConnectedNode):
            identifier = f"Linear_{node_index}"
            new_node = nodes.FullyConnectedNode(identifier, current_node.in_features, current_node.out_features,
                                                current_node.weight, current_node.bias)
            combined_network.add_node(new_node)

        elif isinstance(current_node, nodes.ReLUNode):
            identifier = f"ReLU_{node_index}"
            new_node = nodes.ReLUNode(identifier, current_node.num_features)
            combined_network.add_node(new_node)
        else:
            raise NotImplementedError

        node_index += 1
        current_node = next_node

    if isinstance(current_node, nodes.FullyConnectedNode):
        identifier = f"Linear_{node_index}"
        new_node = nodes.FullyConnectedNode(identifier, current_node.in_features, current_node.out_features,
                                            current_node.weight, current_node.bias)
        combined_network.add_node(new_node)
    elif isinstance(current_node, nodes.ReLUNode):
        identifier = f"ReLU_{node_index}"
        new_node = nodes.ReLUNode(identifier, current_node.num_features)
        combined_network.add_node(new_node)
    else:
        raise NotImplementedError

    return combined_network


def testing(net: cv.PyTorchNetwork, dataset: datasets.Dataset, test_batch_size: int, cuda: bool) -> (float, float):
    """
    Testing procedure for a PyTorchNetwork.

    Parameters
    ----------
    net : PyTorchNetwork
        Neural Network to test.
    dataset : Dataset
        Dataset used for testing the network.
    test_batch_size : int
        Dimension for the test batch size for the testing procedure
    cuda : bool
        Whether to use the cuda library for the procedure (default: False).

    Returns
    ----------
    (float, float)
        Rate of correct samples and loss.

    """

    if cuda:
        net.pytorch_network.cuda()
    else:
        net.pytorch_network.cpu()

    test_set = dataset.get_test_set()

    net.pytorch_network.eval()
    net.pytorch_network.float()
    test_loss = 0
    correct = 0
    with torch.no_grad():

        batch_idx = 0
        data_idx = 0

        while data_idx < len(test_set[0]):

            if data_idx + test_batch_size >= len(test_set[0]):
                last_data_idx = len(test_set[0])
            else:
                last_data_idx = data_idx + test_batch_size

            data = torch.from_numpy(test_set[0][data_idx:last_data_idx, :])
            target = torch.from_numpy(test_set[1][data_idx:last_data_idx])

            if cuda:
                data, target = data.cuda(), target.cuda()

            data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
            output = net.pytorch_network(data)
            test_loss += funct.cross_entropy(output, target, reduction='sum').data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            batch_idx += 1
            data_idx += test_batch_size

    test_loss /= float(math.floor(len(test_set[0]) / test_batch_size))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_set[0]),
        100. * correct / len(test_set[0])))

    return correct / float(len(test_set[0])), test_loss


def generate_targeted_linf_robustness_query(data: Tensor.Tensor, adv_target: int, bounds: tuple,
                                            num_classes: int, epsilon: float, filepath: str):
    """
    Function to generate a targeted Robustness SMTLIB query and to save it to a SMTLIB file.
    The robustness query is of the kind based on the infinity norm.
    It assumes that the data and target are from a classification task.

    Parameters
    ----------
    data : Tensor
        Input data of interest.
    adv_target : int
        Desired adversarial target for the input data.
    bounds : (int, int)
        Bounds for the input data (lower_bound, upper_bound).
    num_classes : int
        Number of possible classes.
    epsilon : float
        Perturbation with respect to the infinity norm.
    filepath : str
        Filepath for the resulting SMTLIB file.

    """
    with open(filepath, "w") as f:

        flattened_data = data.flatten()
        for i in range(len(flattened_data)):
            f.write(f"(declare-const X_{i} Real)\n")

        for i in range(num_classes):
            f.write(f"(declare-const Y_{i} Real)\n")

        for i in range(len(flattened_data)):

            if flattened_data[i] - epsilon < bounds[0]:
                f.write(f"(assert (>= X_{i} {bounds[0]}))\n")
            else:
                f.write(f"(assert (>= X_{i} {flattened_data[i] - epsilon}))\n")

            if flattened_data[i] + epsilon > bounds[1]:
                f.write(f"(assert (<= X_{i} {bounds[1]}))\n")
            else:
                f.write(f"(assert (<= X_{i} {flattened_data[i] + epsilon}))\n")

        for i in range(num_classes):

            if i != adv_target:
                f.write(f"(assert (<= (- Y_{i} Y_{adv_target}) 0))\n")

def generate_untargeted_linf_robustness_query(data: Tensor.Tensor, target: int, bounds: tuple,
                                              num_classes: int, epsilon: float, filepath: str):
    """
    Function to generate an untargeted Robustness SMTLIB query and to save it to a SMTLIB file.
    The robustness query is of the kind based on the infinity norm.
    It assumes that the data and target are from a classification task.

    Parameters
    ----------
    data : Tensor
        Input data of interest.
    adv_target : int
        Desired adversarial target for the input data.
    bounds : (int, int)
        Bounds for the input data (lower_bound, upper_bound).
    num_classes : int
        Number of possible classes.
    epsilon : float
        Perturbation with respect to the infinity norm.
    filepath : str
        Filepath for the resulting SMTLIB file.

    """
    with open(filepath, "w") as f:

        flattened_data = data.flatten()
        for i in range(len(flattened_data)):
            f.write(f"(declare-const X_{i} Real)\n")

        for i in range(num_classes):
            f.write(f"(declare-const Y_{i} Real)\n")

        for i in range(len(flattened_data)):

            if flattened_data[i] - epsilon < bounds[0]:
                f.write(f"(assert (>= X_{i} {bounds[0]}))\n")
            else:
                f.write(f"(assert (>= X_{i} {flattened_data[i] - epsilon}))\n")

            if flattened_data[i] + epsilon > bounds[1]:
                f.write(f"(assert (<= X_{i} {bounds[1]}))\n")
            else:
                f.write(f"(assert (<= X_{i} {flattened_data[i] + epsilon}))\n")

        output_query = "(assert (or"
        for i in range(num_classes):

            if i != target:
                output_query += f" (<= (- Y_{target} Y_{i}) 0)"

        output_query += "))"
        f.write(output_query)


def parse_linf_robustness_smtlib(filepath: str) -> (bool, list, int):
    """
    Function to extract the parameters of a robustness query from the smtlib file.
    It assume the SMTLIB file is structured as following:

        ; definition of the variables of interest
        (declare-const X_0 Real)
        (declare-const X_1 Real)
        ...
        (declare-const Y_1 Real)
        (declare-const Y_2 Real)
        ...
        ; definition of the constraints
        (assert (>= X_0 eps_0))
        (assert (<= X_0 eps_1))
        ...
        (assert (<= (- Y_0 Y_1) 0))
        ...

    Where the eps_i are Real numbers.

    Parameters
    ----------
    filepath : str
        Filepath to the SMTLIB file.

    Returns
    ----------
    (bool, list, int)
        Tuple of list: the first list contains the values eps_i for each variables as tuples (lower_bound, upper_bound),
        while the int correspond to the desired target for the related data.
    """
    targeted = True
    correct_target = -1
    lb = []
    ub = []
    with open(filepath, 'r') as f:

        for line in f:

            line = line.replace('(', '( ')
            line = line.replace(')', ' )')
            if line[0] == '(':
                aux = line.split()
                if aux[1] == 'assert':

                    if aux[4] == '(':
                        if aux[3] == 'or':
                            targeted = False
                            temp = aux[8].split("_")
                            correct_target = int(temp[1])
                        else:
                            targeted = True
                            temp = aux[7].split("_")
                            correct_target = int(temp[1])

                    else:

                        if aux[3] == ">=":
                            lb.append(float(aux[5]))
                        else:
                            ub.append(float(aux[5]))

    input_bounds = []
    for i in range(len(lb)):
        input_bounds.append((lb[i], ub[i]))

    return targeted, input_bounds, correct_target


def net_update(network: networks.NeuralNetwork) -> networks.NeuralNetwork:

    if not network.up_to_date:

        for alt_rep in network.alt_rep_cache:

            if alt_rep.up_to_date:
                if isinstance(alt_rep, cv.ONNXNetwork):
                    return cv.ONNXConverter().to_neural_network(alt_rep)
                elif isinstance(alt_rep, cv.PyTorchNetwork):
                    return cv.PyTorchConverter().to_neural_network(alt_rep)
                else:
                    raise NotImplementedError

    else:
        return network
