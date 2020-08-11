if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import numpy as np

import dezero.functions as F
import dezero.layers as L
from dezero import Variable


class Simple_NN():
    def __init__(self, H, O, seed=0):

        self.l1 = L.Linear(H)
        self.l2 = L.Linear(O)

    def forward(self, x):
        y = self.l1(x)
        y = F.sigmoid_simple(y)
        y = self.l2(y)
        return y


if __name__ == '__main__':

    # variables
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    # size
    H = 10
    O = 1

    nn = Simple_NN(H, O)

    # update

    lr = 0.2
    iters = 10000

    for it in range(iters):
        y_pred = nn.forward(x)
        loss = F.mean_squared_error(y, y_pred)

        nn.l1.cleargrads()
        nn.l2.cleargrads()

        loss.backward()

        for layer in [nn.l1, nn.l2]:
            for p in layer.params():
                p.data -= lr * p.grad.data

        if it % 1000 == 0:
            print(loss)

    plt.scatter(x, y, s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    t = np.arange(0, 1, .01)[:, np.newaxis]
    y_pred = nn.forward(t)
    plt.plot(t, y_pred.data, color='r')
    plt.savefig('step44.png')
