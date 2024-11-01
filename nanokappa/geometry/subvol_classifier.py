import numpy as np
from scipy.interpolate import NearestNDInterpolator


class SubvolClassifier:
    def __init__(self, n, xc=None, a=None):

        self.n = n  # number of subvolumes

        if xc is None:
            self.a = a  # slicing axis
            self.xc = np.ones((self.n, 3)) * 0.5

            # center positions
            self.xc[:, self.a] = np.linspace(0, 1 - 1 / n, n) + 1 / (2 * n)
        else:
            self.xc = xc

        self.f = NearestNDInterpolator(self.xc, np.arange(self.n, dtype=int))

    def predict(self, x):
        return self.f(x).astype(int)
