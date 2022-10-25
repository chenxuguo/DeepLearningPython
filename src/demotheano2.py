import theano
from theano import tensor

x, y = tensor.dmatrices('x', 'y')

diff = x - y

abs_diff = abs(diff)
diff_squared = diff ** 2

f = theano.function([x, y], [diff, abs_diff, diff_squared])

result = f([[1, 1], [1, 1]], [[0, 1], [2, 3]])

print(result[0])

print(result[1])

print(result[2])
