"""Misc utilities"""

import multiprocessing as mp
import signal
import time
from contextlib import contextmanager
from typing import Union

import numpy as np
import pandas as pd


def timed_print(*args):
    """
    Utility function that adds the process name as well as the current
    time to any message.
    """
    mssg = ' '.join([str(a) for a in args])
    s = time.strftime('%Y-%m-%d %H:%M:%S')
    pn = mp.current_process().name
    print("{} -- {}: {}".format(s, pn, mssg))


def print_experiment_times(durations, total):
    """
    prints some info about how long each experiment takes and how long
    it will take to complete all of them

    durations = list with times in seconds of completed experiments
    total = int total number of planned experiments
    """
    mean_duration = np.mean(durations)
    m, s = divmod(mean_duration, 60)
    print("Average duration of one experiment = %02d:%02d" % (m, s))

    exps_left = total - len(durations)
    time_left = exps_left * mean_duration
    m_t, s_t = divmod(time_left, 60)
    if time_left > 3600:
        h_t, m_t = divmod(m_t, 60)
        print("Estimated time left = %02d:%02d:%02d" % (h_t, m_t, s_t))
    else:
        print("Estimated time left = %02d:%02d" % (m_t, s_t))


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    """
    Contextmanager to run a function call for a limited time:

    Example use:

        try:
            with time_limit(10):
                long_function_call()
        except TimeoutException as e:
            print("Timed out!")


    https://stackoverflow.com/questions/366682/
        how-to-limit-execution-time-of-a-function-call-in-python

    """

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def time_limited_df_to_pickle(df: pd.DataFrame, fname: str, t: Union[int, float]):
    """
    Trying twice to save the data frame within the time limit, then giving up.

    """
    try:
        with time_limit(t):
            df.to_pickle(fname)
    except TimeoutException:
        try:
            with time_limit(t):
                df.to_pickle(fname)
        except TimeoutException:
            print(f"Problem saving data:\n"
                  f"File {fname} couldn't be saved in the allocated time!")

def unnormalise_x_given_lims(x_in, lims):
    """
    Scales the input x (assumed to be between [-1, 1] for each dim)
    to the lims of the problem
    """
    # assert len(x_in) == len(lims)

    r = lims[:, 1] - lims[:, 0]
    x_orig = r * (x_in + 1) / 2 + lims[:, 0]

    return x_orig
