if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import numpy as np

import dezero.functions as F
import dezero.layers as L
from dezero import Variable, Model


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


if __name__ == '__main__':

    np.random.seed(0)
    # variables
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    # size
    hidden_size = 10
    out_size = 1

    nn = TwoLayerNet(hidden_size, out_size)

    # update

    lr = 0.2
    iters = 10000

    for it in range(iters):
        y_pred = nn(x)
        loss = F.mean_squared_error(y, y_pred)

        nn.cleargrads()
        loss.backward()

        for p in nn.params():
            p.data -= lr * p.grad.data

        if it % 1000 == 0:
            print(loss)

    plt.scatter(x, y, s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    t = np.arange(0, 1, .01)[:, np.newaxis]
    y_pred = nn.forward(t)
    plt.plot(t, y_pred.data, color='r')
    plt.savefig('step45.png')
