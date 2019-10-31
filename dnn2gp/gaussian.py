import numpy as np
import scipy as sp


class Gaussian:

    def __init__(self, m, S=None, P=None):
        self.m = m
        assert not (S is None and P is None)
        self._S = S
        self._P = P
        self._Scale = None

    @property
    def S(self):
        if self._S is None:
            self._S = np.linalg.pinv(self.P)
        return self._S

    @property
    def P(self):
        if self._P is None:
            self._P = np.linalg.pinv(self.S)
        return self._P

    @property
    def Scale(self):
        if self._Scale is None:
            self._Scale = sp.linalg.sqrtm(self.S)
        return self._Scale

    def sample(self):
        unit_samples = np.random.randn(*self.m.shape)
        return self.m + self.Scale @ unit_samples
