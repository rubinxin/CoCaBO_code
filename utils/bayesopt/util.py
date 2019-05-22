from typing import Tuple

import numpy as np


def add_hallucinations_to_x_and_y(bo, old_x, old_y, x_new, fixed_dim_vals=None) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Add hallucinations to the data arrays.

    Parameters
    ----------
    old_x
        Current x values
    old_y
        Current y values
    x_new
        Locations at which to use the async infill procedure. If x_busy
        is None, then nothing happens and the x and y arrays are returned

    Returns
    -------
    augmented_x (np.ndarray), augmented_y (list or np.ndarray)
    """
    if x_new is None:
        x_out = old_x
        y_out = old_y
    else:
        if isinstance(x_new, list):
            x_new = np.vstack(x_new)

        if fixed_dim_vals is not None:
            if fixed_dim_vals.ndim == 1:  # row vector
                fixed_dim_vals = np.vstack([fixed_dim_vals] * len(x_new))
            assert len(fixed_dim_vals) == len(x_new)
            x_new = np.hstack((
                fixed_dim_vals, x_new
            ))
        x_out = np.vstack((old_x, x_new))
        fake_y = make_hallucinated_data(bo, x_new, bo.async_infill_strategy)
        y_out = np.vstack((old_y, fake_y))

    return x_out, y_out


def make_hallucinated_data(bo, x: np.ndarray,
                           strat: str) -> np.ndarray:
    """Returns fake y-values based on the chosen heuristic

    Parameters
    ----------
    x
        Used to get the value for the kriging believer. Otherwise, this
        sets the number of values returned

    bo
        Instance of BayesianOptimization

    strat
        string describing the type of hallucinated data. Choices are:
        'constant_liar_min', 'constant_liar_median', 'kriging_believer',
        'posterior_simple'

    Returns
    -------
    y : np.ndarray
        Values for the desired heuristic

    """
    if strat == 'constant_liar_min':
        if x is None:
            y = np.atleast_2d(bo.y_min)
        else:
            y = np.array([bo.y_min] * len(x)).reshape(-1, 1)

    elif strat == 'constant_liar_median':
        if x is None:
            y = np.atleast_2d(bo.y_min)
        else:
            y = np.array([bo.y_min] * len(x)).reshape(-1, 1)

    elif strat == 'kriging_believer':
        y = bo.surrogate.predict(x)[0]

    elif strat == 'posterior_simple':
        mu, var = bo.surrogate.predict(x)
        y = np.random.multivariate_normal(mu.flatten(),
                                          np.diag(var.flatten())) \
            .reshape(-1, 1)

    elif strat == 'posterior_full':
        mu, var = bo.surrogate.predict(x, full_cov=True)
        y = np.random.multivariate_normal(mu.flatten(), var).reshape(-1, 1)
    else:
        raise NotImplementedError

    return y
