import numpy as np
from step01 import Variable


class Function():
    def __call__(self, input):
        x = input.data  # pick data
        y = x**2
        output = Variable(y)  # cast
        return output


if __name__ == '__main__':
    x = Variable(np.array(10))
    f = Function()
    y = f(x)
    print(type(y))
    print(y.data)
