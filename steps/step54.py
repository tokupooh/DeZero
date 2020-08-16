if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import test_mode
import dezero.functions as F

if __name__ == '__main__':
    x = np.ones(5)
    print(x)

    # training mode
    y = F.dropout(x)
    print(y)

    #  predicting mode
    with test_mode():
        y = F.dropout(x)
        print(y)
