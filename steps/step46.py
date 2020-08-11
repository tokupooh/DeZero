if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

if __name__ == '__main__':
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    max_iter = 10000
    hidden_size = 10

    model = MLP((hidden_size, 1))
    optimizer = optimizers.Adam().setup(model)
    # optimizer = optimizers.MomentumSGD().setup(model)
    # optimizer = optimizers.SGD().setup(model)
    # optimizer = optimizers.AdaGrad().setup(model)
    # optimizer = optimizers.AdaDelta().setup(model)

    for it in range(max_iter):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        optimizer.update()

        if it % 1000 == 0:
            print(loss)
