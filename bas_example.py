import onnx
from strategies.conversion import ONNXNetwork
from strategies.verification import LocalRobustnessProperty, MarabouVerification
from datasets import MNISTDataset

ds = MNISTDataset()

test_set = ds.get_test_set()

ex_x = test_set[0][10]
ex_y = test_set[1][10]

prop = LocalRobustnessProperty(ex_x, 9, True, 'Linf', 1, [])
verification = MarabouVerification()
print(verification.verify('test_net.onnx', prop))