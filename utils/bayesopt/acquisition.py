# -*- coding: utf-8 -*-
#==========================================
# Title:  acquisition.py
# Author: Binxin Ru and Ahsan Alvi
# Date:   20 August 2019
# Link:   https://arxiv.org/abs/1906.08878
#==========================================

"""
Acquisition functions
"""

import numpy as np
from scipy.stats import norm

from utils.ml_utils.models import GP


class AcquisitionFunction(object):
    """
    Base class for acquisition functions. Used to define the interface
    """

    def __init__(self, surrogate=None, verbose=False):
        self.surrogate = surrogate
        self.verbose = verbose

    def evaluate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError


class AcquisitionOnSubspace():
    def __init__(self, acq, free_idx, fixed_vals):
        self.acq = acq
        self.free_idx = free_idx
        self.fixed_vals = fixed_vals

    def evaluate(self, x: np.ndarray, **kwargs):
        x_fixed = [self.fixed_vals] * len(x)
        x_complete = np.hstack((np.vstack(x_fixed), x))
        return self.acq.evaluate(x_complete)


class EI(AcquisitionFunction):
    """
    Expected improvement acquisition function for a Gaussian model

    Model should return (mu, var)
    """

    def __init__(self, surrogate: GP, best: np.ndarray, verbose=False):
        self.best = best
        super().__init__(surrogate, verbose)

    def __str__(self) -> str:
        return "EI"

    def evaluate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluates the EI acquisition function.

        Parameters
        ----------
        x
            Input to evaluate the acquisition function at

        """
        if self.verbose:
            print("Evaluating EI at", x)
        mu, var = self.surrogate.predict(np.atleast_2d(x))
        var = np.clip(var, 1e-8, np.inf)

        s = np.sqrt(var)
        gamma = (self.best - mu) / s
        return (s * gamma * norm.cdf(gamma) + s * norm.pdf(gamma)).flatten()


class PI(AcquisitionFunction):
    """
    Probability of improvement acquisition function for a Gaussian model

    Model should return (mu, var)
    """

    def __init__(self, surrogate: GP, best: np.ndarray, tradeoff: float,
                 verbose=False):
        self.best = best
        self.tradeoff = tradeoff

        super().__init__(surrogate, verbose)

    def __str__(self) -> str:
        return f"PI-{self.tradeoff}"

    def evaluate(self, x, **kwargs) -> np.ndarray:
        """
        Evaluates the PI acquisition function.

        Parameters
        ----------
        x
            Input to evaluate the acquisition function at

        """
        if self.verbose:
            print("Evaluating PI at", x)
        mu, var = self.surrogate.predict(x)
        var = np.clip(var, 1e-8, np.inf)

        s = np.sqrt(var)
        gamma = (self.best - mu - self.tradeoff) / s
        return norm.cdf(gamma).flatten()


class UCB(AcquisitionFunction):
    """
    Upper confidence bound acquisition function for a Gaussian model

    Model should return (mu, var)
    """

    def __init__(self, surrogate: GP, tradeoff: float, verbose=False):
        self.tradeoff = tradeoff

        super().__init__(surrogate, verbose)

    def __str__(self) -> str:
        return f"UCB-{self.tradeoff}"

    def evaluate(self, x, **kwargs) -> np.ndarray:
        """
        Evaluates the UCB acquisition function.

        Parameters
        ----------
        x
            Input to evaluate the acquisition function at
        """
        if self.verbose:
            print("Evaluating UCB at", x)
        mu, var = self.surrogate.predict(x)
        var = np.clip(var, 1e-8, np.inf)

        s = np.sqrt(var)  # type: np.ndarray
        return -(mu - self.tradeoff * s).flatten()
