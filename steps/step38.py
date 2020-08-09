if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

from dezero import Variable

if __name__ == '__main__':

    x = Variable(np.array([[0, 1, 2], [3, 4, 5]]))
    y = x.reshape((6, ))
    y.backward(retain_grad=True)
    print(x.grad)

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = x.T
    y.backward()
    print(x.grad)
