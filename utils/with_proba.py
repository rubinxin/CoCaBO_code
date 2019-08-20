# -*- coding: utf-8 -*-
#==========================================
# Title:  with_proba.py
# Author: Binxin Ru and Ahsan Alvi
# Date:   20 August 2019
# Link:   https://arxiv.org/abs/1906.08878
#==========================================


""" Simply defines a function :func:`with_proba` that is used everywhere.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = ""
__version__ = ""

from random import random


# --- Utility functions
def with_proba(epsilon):
    """Bernoulli test, with probability :math:`\varepsilon`, return `True`, and with probability :math:`1 - \varepsilon`, return `False`.

    Example:

    >>> from random import seed; seed(0)  # reproductible
    >>> with_proba(0.5)
    False
    >>> with_proba(0.9)
    True
    >>> with_proba(0.1)
    False
    >>> if with_proba(0.2):
    ...     print("This happens 20% of the time.")
    """
    assert 0 <= epsilon <= 1, "Error: for 'with_proba(epsilon)', epsilon = {:.3g} has to be between 0 and 1 to be a valid probability.".format(epsilon)  # DEBUG
    return random() < epsilon  # True with proba epsilon

