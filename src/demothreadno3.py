import numpy
import theano
from theano import tensor
from theano import pp

x = tensor.dmatrix('x')

s = tensor.sum(1 / (1 + tensor.exp(-x)))

gs = tensor.grad(s, x)

dlogistic = theano.function([x], gs)

print(dlogistic([[0, 1], [-1, -2]]))
