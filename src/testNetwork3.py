# ----------------------
# - network3.py example:
import network3
from network3 import Network, ConvPoolLayer, FullyConnectedLayer, \
    SoftmaxLayer  # softmax plus log-likelihood cost is more common in modern image classification networks.

# read data:
training_data, validation_data, test_data = network3.load_data_shared()
# mini-batch size:
mini_batch_size = 10

# chapter 6 - shallow architecture using just a single hidden layer, containing 100 hidden neurons.

# net = Network([
#     FullyConnectedLayer(n_in=784, n_out=100),
#     SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
# net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)

# chapter 6 - 5x5 local receptive fields, 20 feature maps, max-pooling layer 2x2

net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2)),
    FullyConnectedLayer(n_in=20*12*12, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)

