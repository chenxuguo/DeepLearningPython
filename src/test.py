"""
Testing code for different neutral network configurations
Adapted for pyth 3.5.2

Usage in shell:
   python3.5 test.py

Network (network.py and network2.py) parameters
    2nd params in epochs count
    3rd param is batch size
    4th param is learning rate (eta)

Author:
    Michaf Dobrzariski, 2016
    dobrzanski.michai.daniel@gmail.com
"""
import mnist_loader
import network
import network2

# ----------------
# - read the input data:
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

# --------------
# - network.py example:
# net = network.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# ------------
# - network2.py example:

# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# # net.large_weight_initializer()
# net.SGD(training_data, 30, 10, 0.1, lmbda=5.0, evaluation_data=validation_data,
#      monitor_evaluation_accuracy=True)

# chapter 3 - Overfitting example - too many epochs of learning applied on small (1k samples) amount od data.
# Overfitting is treating noise as a signal.
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data,
#        monitor_evaluation_accuracy=True,
#        monitor_training_cost=True)

# chapter3 - Regularization(weight decay)example 1(ony 100 of training data and 30 hidden neurons)
"""
net = network2.Network([784, 30, 10], cost = network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data[:1000], 400, 10, 0.5,
        evaluation_data=test_data,
        lmbda=0.1,  # this is a regularization parameter
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
"""
# chapter 3 - Early stopping implemented
"""
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data[:1000], 30, 10, 0.5,
        lmbda=5.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        early_stopping_n=10)
"""

# chapter 4 - The vanishing gradient problem - deep networks are hard to train with simple SGD algorithm
# this network learns much slower than a shallow one.

net = network2.Network([784, 30, 30, 30, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.1,
        lmbda=5.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True)

