if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np

import dezero
import dezero.functions as F
import dezero.layers as L
from dezero import Model


class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y


if __name__ == '__main__':

    # data
    np.random.seed(0)
    seq_data = [np.random.randn(1, 1) for _ in range(1000)]
    xs = seq_data[0:-1]
    ts = seq_data[1:]

    # parameters
    max_epoch = 100
    hidden_size = 100
    bptt_lengh = 30

    train_set = dezero.datasets.SinCurve(train=True)
    seqlen = len(train_set)

    model = SimpleRNN(hidden_size, 1)
    optimizer = dezero.optimizers.Adam().setup(model)

    for epoch in range(max_epoch):
        model.reset_state()
        loss = 0
        count = 0
        for x, t in train_set:
            x = x.reshape(1, 1)
            y = model(x)
            loss += F.mean_squared_error(y, t)
            count += 1

            if count % bptt_lengh == 0 or count == seqlen:
                model.cleargrads()
                loss.backward()
                loss.unchain_backward()
                optimizer.update()
        avg_loss = float(loss.data) / count
        print(f'| epoch {epoch+1} | loss {avg_loss :.3f}')

    # Plot
    xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
    model.reset_state()
    pred_list = []

    with dezero.no_grad():
        for x in xs:
            x = np.array(x).reshape(1, 1)
            y = model(x)
            pred_list.append(float(y.data))

    plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
    plt.plot(np.arange(len(xs)), pred_list, label='predict')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('step59.png', dpi=300)
