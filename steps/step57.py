if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time

import numpy as np

import dezero.functions as F
from dezero import Variable

if __name__ == '__main__':
    np.random.seed(0)
    x1 = np.random.rand(1, 3, 7, 7)  # one data
    print(f'x1: {x1.shape}')
    col1 = F.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
    print(f'col1: {col1.shape}')

    x2 = np.random.rand(10, 3, 7, 7)  # ten data
    print(f'x2: {x2.shape}')
    col2 = F.im2col(x2, kernel_size=5, stride=1, pad=0, to_matrix=True)
    print(f'col2: {col2.shape}')

    # conv2d
    N, C, H, W = 10, 5, 100, 100
    OC, KH, KW = 8, 3, 3

    x = Variable(np.random.randn(N, C, H, W))
    W = np.random.randn(OC, C, KH, KW)

    start = time.time()
    y = F.conv2d(x, W, b=None, stride=1, pad=1)
    y.backward()

    elapsed_time = time.time() - start
    print(f'elapsed time: {elapsed_time} [sec]')
    print(f'y shape: {y.shape}')

    print(f'x grad: {x.grad.shape}')
