"""Temporal smoothing for the joint angles."""
import numpy as np


class OneEuro:
    """One-Euro filter (Casiez et al. 2012): smooths hard while the signal is still, follows quickly when it moves.
    mc = minimum cut-off in Hz (lower = smoother, laggier); beta = how fast the cut-off opens up with speed."""

    def __init__(self, mc=0.7, beta=0.03, dc=1.0):
        self.mc, self.beta, self.dc, self.x, self.dx, self.t = mc, beta, dc, None, 0.0, None

    @staticmethod
    def _a(cut, dt):
        return 1.0 / (1.0 + 1.0 / (2 * np.pi * cut * dt))

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.t = x, t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        self.dx += self._a(self.dc, dt) * ((x - self.x) / dt - self.dx)
        self.x += self._a(self.mc + self.beta * abs(self.dx), dt) * (x - self.x)
        return self.x
