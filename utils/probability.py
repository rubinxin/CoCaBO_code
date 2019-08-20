# -*- coding: utf-8 -*-
#==========================================
# Title:  probability.py
# Author: Binxin Ru and Ahsan Alvi
# Date:   20 August 2019
# Link:   https://arxiv.org/abs/1906.08878
#==========================================

"""
Created based on
https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/
"""

import random


# draw: [float] -> int
# pick an index from the given list of floats proportionally
# to the size of the entry (i.e. normalize to a probability
# distribution and draw according to the probabilities).
def draw(weights):
    choice = random.uniform(0, sum(weights))
    #    print(choice)
    choiceIndex = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex
        choiceIndex += 1


# distr: [float] -> (float)
# Normalize a list of floats to a probability distribution.  Gamma is an
# egalitarianism factor, which tempers the distribtuion toward being uniform as
# it grows from zero to one.
def distr(weights, gamma=0.0):
    theSum = float(sum(weights))
    return tuple((1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights)


def mean(aList):
    theSum = 0
    count = 0

    for x in aList:
        theSum += x
        count += 1

    return 0 if count == 0 else theSum / count
