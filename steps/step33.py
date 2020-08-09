if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable


def f(x):
    y = x**4 - 2 * x**2
    return y


if __name__ == '__main__':
    x = Variable(np.array(2.0))
    iters = 10

    for it in range(iters):
        print(f'Iter: {it}, x: {x.data}')

        y = f(x)
        x.cleargrad()
        y.backward(create_graph=True)

        gx = x.grad
        # ここでxの勾配をクリアしないとx.gradの値が残った状態で微分を計算してしまうので
        # リセットする必要がある
        x.cleargrad()
        gx.backward()
        gx2 = x.grad

        x.data -= gx.data / gx2.data

    print('Finish!!')
    print(f'x: {x.data}')
