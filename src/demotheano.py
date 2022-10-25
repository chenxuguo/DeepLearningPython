import theano
from theano import tensor

# declare a variable
x = tensor.dmatrix('x')

# create the expression
s = 1 / (1 + tensor.exp(-x))

#
logistic = theano.function([x], s)

print(logistic([[0, 1], [-1, -2]]))
