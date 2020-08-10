if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import numpy as np

import dezero.functions as F
from dezero import Variable


class Simple_NN():
    def __init__(self, I, H, O, seed=0):
        self.I = I
        self.H = H
        self.O = O

        np.random.seed(seed)
        self.W1 = Variable(0.01 * np.random.randn(self.I, self.H))
        self.b1 = Variable(np.zeros(self.H))
        self.W2 = Variable(0.01 * np.random.randn(self.H, self.O))
        self.b2 = Variable(np.zeros(self.O))

    def forward(self, x):
        y = F.linear(x, self.W1, self.b1)
        y = F.sigmoid_simple(y)
        y = F.linear(y, self.W2, self.b2)
        return y


if __name__ == '__main__':

    # variables
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    # size
    I = 1
    H = 10
    O = 1

    nn = Simple_NN(I, H, O)

    # update

    lr = 0.2
    iters = 10000

    for it in range(iters):
        y_pred = nn.forward(x)
        loss = F.mean_squared_error(y, y_pred)

        nn.W1.cleargrad()
        nn.b1.cleargrad()
        nn.W2.cleargrad()
        nn.b2.cleargrad()

        loss.backward()

        nn.W1.data -= lr * nn.W1.grad.data
        nn.b1.data -= lr * nn.b1.grad.data
        nn.W2.data -= lr * nn.W2.grad.data
        nn.b2.data -= lr * nn.b2.grad.data
        if it % 1000 == 0:
            print(loss)

    plt.scatter(x, y, s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    t = np.arange(0, 1, .01)[:, np.newaxis]
    y_pred = nn.forward(t)
    plt.plot(t, y_pred.data, color='r')
    plt.savefig('step43.png')
