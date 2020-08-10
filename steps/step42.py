if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import numpy as np

import dezero.functions as F
from dezero import Variable


def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff**2) / len(diff)


def predict(x):
    y = F.matmul(x, W) + b
    return y


if __name__ == '__main__':
    np.random.seed(1)
    x = np.random.rand(100, 1)
    y = 5 + 2 * x + np.random.rand(100, 1)
    x = Variable(x)
    y = Variable(y)

    W = Variable(np.zeros((1, 1)))
    b = Variable(np.zeros(1))

    lr = 0.1
    iters = 100

    for it in range(iters):
        y_pred = predict(x)
        loss = mean_squared_error(y, y_pred)

        W.cleargrad()
        b.cleargrad()
        loss.backward()

        W.data -= lr * W.grad.data
        b.data -= lr * b.grad.data

    plt.scatter(x.data, y.data, s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    y_pred = predict(x)

    plt.plot(x.data, y_pred.data, color='r')
    plt.savefig('step42.png')
