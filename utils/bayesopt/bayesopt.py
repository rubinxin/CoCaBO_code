# -*- coding: utf-8 -*-
#==========================================
# Title:  bayesopt.py
# Author: Binxin Ru and Ahsan Alvi
# Date:   20 August 2019
# Link:   https://arxiv.org/abs/1906.08878
#==========================================

import os
import time
from typing import Tuple, Callable, Optional, Dict, Union

import numpy as np
import pandas as pd
import pylab as plt

from utils.bayesopt.acquisition import AcquisitionOnSubspace
from .acquisition import AcquisitionFunction, EI, PI, UCB
from utils.ml_utils import timed_print as print
from utils.ml_utils.models import GP
from utils.ml_utils.optimization import minimize_with_restarts, sample_then_minimize


class BayesianOptimisation(object):
    """Bayesian Optimisation class

    Parameters
    ----------
    sampler : Callable
        function handle returning sample from expensive function being
        optimized

    surrogate : basic_gp.GP
        (GP) model that models the surface of 'objective'

    bounds : ndarray
        bounds of each dimension of x as a Dx2 vector (default [0, 1])

    batch_size : int
        How many tasks to suggest in one go.

    acq_dict : dict
        Defaults to EI

    optimise_surrogate_model : bool
        Whether to optimise the surrogate model after each BayesOpt iteration

    track_cond_k : bool
        Whether to keep track of cond(K) of the surrogate model across
        BayesOpt iterations

    y_min_opt_params : dict
        opt_params dict with the following fields:

            - method = 'standard', multigrad', 'direct'
            - n_direct_evals = for direct
            - num_restarts = for multigrad

    acq_opt_params : dict
        opt_params dict with the following fields:

            - method = 'multigrad', 'direct'
            - n_direct_evals = for direct
            - num_restarts = for multigrad

    n_bo_steps : int
        Number of BayesOpt steps

    min_acq : float
        cut-off threshold for acquisition function

    """

    def __init__(self, sampler: Callable, surrogate: GP, bounds: np.ndarray,
                 batch_size: int = 1,
                 acq_dict: Optional[Dict] = None,
                 y_min_opt_params: Optional[dict] = None,
                 acq_opt_params: Optional[dict] = None,
                 n_bo_steps: Optional[int] = 30,
                 optimise_surrogate_model: Optional[bool] = True,
                 optimize_every_n_data: Optional[int] = None,
                 optimize_every_n_iter: Optional[int] = None,
                 track_cond_k: Optional[bool] = True,
                 min_acq: Optional[float] = None,
                 create_plots: Optional[bool] = False,
                 save_plots: Optional[bool] = False,
                 plots_prefix: Optional[str] = False,
                 verbose: Optional[Union[bool, int]] = False,
                 debug: Optional[bool] = False,
                 **kwargs):
        self.verbose = verbose
        self.debug = debug
        self.df = None
        self.curr_bo_step = None
        self.batch_size = batch_size

        if self.verbose:
            print("Initialising BayesOpt instance...")

        if acq_dict is None:
            self.acq_dict = {'type': 'EI'}
        else:
            self.acq_dict = acq_dict
        self.min_acq = min_acq

        if y_min_opt_params is None:
            self.y_min_opt_params = {'method': 'direct',
                                     'n_direct_evals': 100}
        else:
            self.y_min_opt_params = y_min_opt_params

        if acq_opt_params is None:
            self.acq_opt_params = {'method': 'direct',
                                   'n_direct_evals': 100}
        else:
            self.acq_opt_params = acq_opt_params

        self.n_bo_steps = n_bo_steps
        self.param_array_hist = []

        self.surrogate = surrogate
        self.sampler = sampler
        self.bounds = bounds

        if optimize_every_n_data is not None:
            self.optimise_surrogate_model_flag = 'data'
            self.optimize_every_n_data = optimize_every_n_data
            self.opt_next_at_n_data = \
                len(self.surrogate.Y) + self.optimize_every_n_data
        elif optimize_every_n_iter is not None:
            raise NotImplementedError  # not tested yet
            self.optimise_surrogate_model_flag = 'iter'
            self.optimize_every_n_iter = optimize_every_n_iter
            self.opt_next_at_iter = None
        elif optimise_surrogate_model:
            self.optimise_surrogate_model_flag = 'always'
        self.counter_since_last_surrogate_opt = 0

        self.x_min = None
        self.y_min = None
        self.var_at_y_min = None

        self.acq_hist = np.zeros(self.n_bo_steps)
        self.sampling_locs = np.zeros((self.n_bo_steps, self.bounds.shape[0]))

        # For debugging numerical issues
        self.track_cond_k = track_cond_k
        if self.track_cond_k:
            self.cond_k_hist = np.zeros(self.n_bo_steps)
        else:
            self.cond_k_hist = None

        # Keep a record of the best sample so far
        self.y_min_hist = np.zeros(self.n_bo_steps)
        self.var_at_y_min_hist = np.zeros(self.n_bo_steps)
        self.x_min_hist = np.zeros((self.n_bo_steps, self.bounds.shape[0]))

        self.create_plots = create_plots
        self.save_plots = save_plots
        self.plots_prefix = plots_prefix

        if self.verbose:
            print("Initialisation of BayesOpt instance done.")

    def _create_acq_function(self, surrogate=None, acq_dict=None,
                             **kwargs) -> AcquisitionFunction:
        """Create the acquisition function object

        This functionality is used, as we then have more flexibility.
        Previously, the acquisition function provided at the start was
        always used and parameters were passed in the evaluate() method,
        which was clunky

        Parameters
        ----------
        surrogate
            Optional: The surrogate model to be used in the acquisition func

        Returns
        -------
        AcquisitionFunction
            The instantiation of the desired acquisition function class
        """
        if surrogate is None:
            surrogate = self.surrogate

        if acq_dict is None:
            acq_dict = self.acq_dict

        if acq_dict['type'] == 'EI':
            return EI(surrogate, self.y_min, **kwargs)
        elif acq_dict['type'] == 'PI':
            tradeoff = acq_dict['tradeoff']
            return PI(surrogate, self.y_min, tradeoff, **kwargs)
        elif acq_dict['type'] == 'UCB':
            tradeoff = acq_dict['tradeoff']
            return UCB(surrogate, tradeoff, **kwargs)
        elif acq_dict['type'] == 'MES':
            f_mins_samples = sample_fmin_Gumble(surrogate, self.bounds,
                                                nMs=acq_dict['n_samples'])
            surrogate.fmin_samples = f_mins_samples
            return MES(surrogate, **kwargs)
        elif acq_dict['type'] == 'BALD':
            return BALD(surrogate, **kwargs)
        elif acq_dict['type'] == 'Random':
            return RandomAcq()
        elif acq_dict['type'] == 'EBALD':  # exact BALD
            raise NotImplementedError
        elif acq_dict['type'] == 'subspace':
            if 'acq_dict' in acq_dict.keys():
                inner_dict = acq_dict['acq_dict']
            else:
                inner_dict = {'type': 'UCB',
                              'tradeoff': 2.0}

            inner_acq = self._create_acq_function(surrogate=surrogate,
                                                  acq_dict=inner_dict)
            fixed_vals = surrogate.fixed_dim_vals
            free_idx = np.arange(len(fixed_vals), surrogate.kern.input_dim)
            acq = AcquisitionOnSubspace(inner_acq, free_idx=free_idx,
                                        fixed_vals=fixed_vals)
            return acq
        else:
            raise NotImplementedError

    def _optimise_acq_func(self, acq, max_or_min='max', acq_opt_params=None):
        """
        Run the chosen optimisation procedure
        """

        if self.verbose:
            print(f"Optimising acquisition function ({max_or_min})")

        if acq_opt_params is None:
            acq_opt_params = self.acq_opt_params

        if max_or_min == 'max':
            def optimiser_func(x):
                return -acq.evaluate(np.atleast_2d(x))
        elif max_or_min == 'min':
            def optimiser_func(x):
                return acq.evaluate(np.atleast_2d(x))
        else:
            raise NotImplementedError

        if acq_opt_params['method'] == 'direct':
            n_direct_evals = acq_opt_params['n_direct_evals']
            res = scipydirect.minimize(optimiser_func,
                                       self.bounds,
                                       maxf=n_direct_evals)

        elif acq_opt_params['method'] == 'multigrad':
            num_restarts = acq_opt_params['num_restarts']
            if 'minimize_options' in acq_opt_params.keys():
                minimize_options = acq_opt_params['minimize_options']
            else:
                minimize_options = None

            res = minimize_with_restarts(optimiser_func,
                                         self.bounds,
                                         num_restarts=num_restarts,
                                         hard_bounds=self.bounds,
                                         minimize_options=minimize_options,
                                         verbose=False)

        elif acq_opt_params['method'] == 'samplegrad':
            if 'minimize_options' in acq_opt_params.keys():
                minimize_options = acq_opt_params['minimize_options']
            else:
                minimize_options = None

            if 'num_samples' in acq_opt_params.keys():
                num_samples = acq_opt_params['num_samples']
            else:
                num_samples = 1000
            if 'num_local' in acq_opt_params.keys():
                num_local = acq_opt_params['num_local']
            else:
                num_local = 5
            if 'num_chunks' in acq_opt_params.keys():
                num_chunks = acq_opt_params['num_chunks']
            else:
                num_chunks = 5
            if 'evaluate_sequentially' in acq_opt_params.keys():
                evaluate_sequentially = \
                    acq_opt_params['evaluate_sequentially']
            else:
                evaluate_sequentially = True

            res = sample_then_minimize(
                optimiser_func,
                self.bounds,
                num_samples=num_samples,
                num_local=num_local,
                num_chunks=num_chunks,
                minimize_options=minimize_options,
                evaluate_sequentially=evaluate_sequentially,
                verbose=False)
        else:
            raise NotImplementedError

        best_x = np.atleast_2d(res.x)

        # if isinstance(acq, AcquisitionOnSubspace):
        #     best_x = np.hstack((np.atleast_2d(acq.fixed_vals),
        #                         np.atleast_2d(best_x)))

        # Return the correct value for the acquisition function depending on
        # whether we minimized or maximized
        if max_or_min == 'max':
            best_eval = best_x, -res.fun
        elif max_or_min == 'min':
            best_eval = best_x, res.fun
        else:
            raise NotImplementedError

        return best_eval

    def get_next(self):
        """Finds the next point to sample at

        Returns
        -------
        x_best : np.ndarray
            Location to sample at

        acq_at_x_best : float
            Value of the acquisition function at the sampling locations

        """
        assert self.batch_size == 1, "Batch BO not yet implemented..."

        acq = self._create_acq_function()
        best_eval = self._optimise_acq_func(acq)
        x_best, acq_at_x_best = best_eval

        if self.verbose > 1:
            print("Optimised acq_dict function")
            print("x, acq_dict(x) = {}".format(best_eval))

        mu, _ = self.surrogate.predict(x_best)

        return x_best, acq_at_x_best

    def _update_surrogate_with_new_data(self, new_sample_x, new_sample_y):

        if self.verbose > 1:
            print("Updating the surrogate model's data arrays")
        model_x, model_y = self.surrogate.X, self.surrogate.Y_raw
        new_x = np.vstack((model_x, new_sample_x))
        new_y = np.vstack((model_y, new_sample_y))
        self.surrogate.set_XY(X=new_x, Y=new_y)

    def run(self):
        """
        Run the BayesOpt loop
        """
        t0 = time.time()
        if self.verbose:
            print("Started BayesOpt.run()")

        self._initialise_bo_df()

        for self.curr_bo_step in range(0, self.n_bo_steps):
            # try:
            if True:
                # if True:
                t1 = time.time()
                if self.verbose:
                    print("**--** Starting BayesOpt iteration {}/{} **--**"
                          .format(self.curr_bo_step + 1, self.n_bo_steps))

                self.optimize_surrogate_if_needed()

                self.x_min, self.y_min, self.var_at_y_min = self._get_y_min()
                x_best, acq_at_x_best = self.get_next()

                t2 = time.time()
                time_taken = t2 - t1

                if self.create_plots:
                    self.plot_step(x_best=x_best)

                # get new y
                new_sample_x, new_sample_y = self._sample_at_x(x_best)
                self._update_surrogate_with_new_data(new_sample_x,
                                                     new_sample_y)

                self.save_history(acq_at_x_best)

                # update variables for plotting
                self.sampling_locs[self.curr_bo_step] = x_best

                self._update_bo_df(x_best, acq_at_x_best, new_sample_x,
                                   new_sample_y, time_taken)

                if self.curr_bo_step == self.n_bo_steps - 1:  # last step
                    if self.verbose > 1:
                        print("Used up budget.")
                        print("Minimum at",
                              self.surrogate.X[np.argmin(self.surrogate.Y)])

            # except np.linalg.linalg.LinAlgError:
            #     print("WARNING: BayesOpt crashed at iteration {}!"
            #           .format(self.curr_bo_step))
            #     break
        if self.verbose:
            print(f"Completed BO exp in {round(time.time() - t0, 2)}s")

    def save_history(self, acq_at_x_best):
        # old variables for keeping track of the experiment. Will
        # remove soon.
        self.acq_hist[self.curr_bo_step] = acq_at_x_best
        self.x_min_hist[self.curr_bo_step] = self.x_min
        self.y_min_hist[self.curr_bo_step] = self.y_min
        self.var_at_y_min_hist[self.curr_bo_step] = self.var_at_y_min
        self.param_array_hist.append(self.surrogate.param_array)

        if self.track_cond_k:
            self.cond_k_hist[self.curr_bo_step] = np.linalg.cond(
                self.surrogate.Ka)

    def _update_bo_df(self, x_best, acq_at_x_best, new_sample_x, new_sample_y,
                      time_taken):
        """Updates the local dataframe with the current iteration's data

        Parameters
        ----------
        x_best
            Best location to sample at

        acq_at_x_best
            Acquisition function value at x_best

        new_sample_x
            actual sample received

        new_sample_y
            actual sample received

        time_taken
            time taken for the iteration in seconds

        """
        if type(self.y_min) == np.ndarray:
            y_min = self.y_min.item()
        else:
            y_min = self.y_min
        current_record = {'ii': self.curr_bo_step,
                          'iteration': self.curr_bo_step + 1,
                          'y_min': y_min,
                          'x_min': self.x_min,
                          'n_data': len(self.surrogate.X),
                          'model_x': self.surrogate.X,
                          'model_y': self.surrogate.Y,
                          'model_param_array': self.surrogate.param_array,
                          'acq_at_sample': acq_at_x_best,
                          'requested_x_sample': x_best,
                          'y_sample': new_sample_y,
                          'x_sample': new_sample_x,
                          'runtime': time_taken,
                          'var_at_y_min': self.var_at_y_min,
                          'cond_k': (self.cond_k_hist[
                                         self.curr_bo_step] if
                                     self.track_cond_k else None)}
        current_record = self.add_info_to_record(current_record)
        self.df = self.df.append([current_record], sort=True)

    def add_info_to_record(self, record, starting=False) -> Dict:
        """
        Used by subclasses to save more info.

        Starting = True is for the starting record
        """
        if starting:
            return record
        else:
            return record

    def _initialise_bo_df(self):
        # self.df = pd.DataFrame(columns=['ii',
        #                                 'y_min',
        #                                 'x_min',
        #                                 'n_data',
        #                                 'model_x',
        #                                 'model_y',
        #                                 'model_param_array',
        #                                 'acq_at_sample',
        #                                 'requested_x_sample',
        #                                 'x_sample',
        #                                 'y_sample',
        #                                 'runtime',
        #                                 'var_at_y_min',
        #                                 'cond_k'])
        self.x_min, self.y_min, self.var_at_y_min = self._get_y_min()
        starting_record = {'ii': -1,
                           'iteration': 0,
                           'y_min': self.y_min,
                           'x_min': self.x_min,
                           'n_data': len(self.surrogate.X),
                           'model_x': self.surrogate.X,
                           'model_y': self.surrogate.Y_raw,
                           'model_param_array': self.surrogate.param_array,
                           'acq_at_sample': np.nan,
                           'requested_x_sample': np.nan,
                           'y_sample': np.nan,
                           'x_sample': np.nan,
                           'runtime': np.nan,
                           'var_at_y_min': self.var_at_y_min,
                           'cond_k': np.nan}
        starting_record = self.add_info_to_record(starting_record,
                                                  starting=True)
        # self.df = self.df.append([starting_record], sort=True)
        self.df = pd.DataFrame.from_dict([starting_record])

    def _sample_at_x(self, x_best) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the sampler object at these points.

        This function is useful if we want to impose specific noise
        on the measurements

        Parameters
        ----------
        x_best
            Locations to sample at

        Returns
        -------
        x : np.ndarray
            Sampled locations x (this is useful in case of input noise)
        y : np.ndarray
            Values of y at the sampled locations

        """
        if self.verbose > 1:
            print("Sampling at selected location")
        new_sample = self.sampler(x_best)
        # new_sample_x = new_sample['x_sample']
        # new_sample_y = new_sample['f_sample']
        # return new_sample_x, new_sample_y
        return x_best, new_sample

    def _get_y_min(self):
        """
        Get y_min for EI computation

        Returns smallest y_min of the model. Can change this to evaluate
        lowest y over domain later.
        """
        if self.verbose:
            print("Finding y_min")

        def optimiser_func(x):
            return self.surrogate.predict(np.atleast_2d(x))[0].flatten()

        if self.y_min_opt_params['method'] == 'standard':
            idx = np.argmin(self.surrogate.Y_raw)
            x_min = self.surrogate.X[idx]
            y_min = self.surrogate.Y_raw[idx]
        elif self.y_min_opt_params['method'] == 'direct':
            def optimiser_func(x):
                return self.surrogate.predict(np.array([x]))[0]

            n_direct_evals = self.y_min_opt_params['n_direct_evals']
            res = scipydirect.minimize(optimiser_func,
                                       self.bounds,
                                       maxf=n_direct_evals)
            x_min = res.x
            y_min = res.fun
        elif self.y_min_opt_params['method'] == 'multigrad':

            num_restarts = self.y_min_opt_params['num_restarts']
            res = minimize_with_restarts(optimiser_func, self.bounds,
                                         num_restarts=num_restarts,
                                         verbose=False)
            x_min = res.x
            y_min = res.fun

        elif self.y_min_opt_params['method'] == 'samplegrad':
            op = self.y_min_opt_params

            if 'minimize_options' in op.keys():
                minimize_options = op['minimize_options']
            else:
                minimize_options = None

            if 'num_samples' in op.keys():
                num_samples = op['num_samples']
            else:
                num_samples = 1000
            if 'num_local' in op.keys():
                num_local = op['num_local']
            else:
                num_local = 5
            if 'evaluate_sequentially' in op.keys():
                evaluate_sequentially = \
                    op['evaluate_sequentially']
            else:
                evaluate_sequentially = False

            res = sample_then_minimize(
                optimiser_func,
                self.bounds,
                num_samples=num_samples,
                num_local=num_local,
                minimize_options=minimize_options,
                evaluate_sequentially=evaluate_sequentially,
                extra_locs=self.surrogate.X,
                verbose=False)

            x_min = res.x
            y_min = res.fun

        else:
            raise NotImplementedError

        _, var_at_y_min = self.surrogate.predict(np.atleast_2d(x_min))

        if self.verbose:
            print(f"Current y_min = {y_min}")

        return x_min, y_min.item(), var_at_y_min.item()

    def plot_y_min(self):
        plt.plot(np.arange(len(self.y_min_hist)), self.y_min_hist)

    def plot_step(self, x_best=None, save_plots=None, external_call=False,
                  **kwargs):
        """
        Plots a summary of the BayesOpt step and saves the image to a
        specified folder
        """
        if save_plots is None:
            save_plots = self.save_plots

        text = "Task"
        figsize = (12, 9)

        acq = self._create_acq_function()

        # Get the best EI/ best minimum
        # if x_best is None:
        #     best_eval = self._optimise_acq_func(acq)
        #     x_best = best_eval[0]
        p = x_best

        if len(self.bounds) == 1:  # 1D
            n_x = 100
            x_dense = np.linspace(self.bounds[0, 0], self.bounds[0, 1], n_x)

            # get mean and std over the domain
            f_mean, f_var = self.surrogate.predict(x_dense[:, None])
            f_mean = f_mean.flatten()
            f_var = f_var.flatten()
            f_std = np.sqrt(f_var)

            acq_dense = acq.evaluate(x_dense[:, None])

            if self.verbose:
                print("Preparing plots")

            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize,
                                     sharex='all')

            # Mean and variance of surrogate
            axes[0].plot(x_dense, f_mean, 'k', label='Surrogate')
            axes[0].fill_between(x_dense,
                                 f_mean + 2 * f_std,
                                 f_mean - 2 * f_std, color='b', alpha=0.2)
            axes[0].set_title(f'Step {self.curr_bo_step} Surrogate model')

            # Data in the surrogate
            if len(self.surrogate.X) > 0:
                axes[0].plot(self.surrogate.X, self.surrogate.Y_raw, 'b*',
                             markersize=16, label='Data')

            # New point(s)
            if x_best is not None:
                axes[0].plot(x_best, self.surrogate.predict(x_best)[0], 'r*',
                             markersize=16, label='New')

            # Acquisition function
            axes[1].plot(x_dense, acq_dense, 'k')

            # New point(s)
            if x_best is not None:
                axes[1].plot(x_best, acq.evaluate(x_best), 'r*',
                             markersize=16, label='New')

            axes[1].set_title('Acquisition function')

            if not external_call:
                axes[0].legend(numpoints=1)
                axes[1].legend(numpoints=1)

        elif len(self.bounds) == 2:  # 2D
            n_x1, n_x2 = 20, 20
            x1 = np.linspace(self.bounds[0, 0], self.bounds[0, 1], n_x1)
            x2 = np.linspace(self.bounds[1, 0], self.bounds[1, 1], n_x2)[::-1]

            x1x2 = np.dstack(np.meshgrid(x1, x2)).reshape(-1, 2)

            y_grid = self.sampler(x1x2)['f_of_x']

            # get mean and std over the domain
            f_mean, f_var = self.surrogate.predict(x1x2)
            f_mean = f_mean[:, 0]
            f_var = f_var[:, 0]
            f_std = np.sqrt(f_var)

            acq_dense = acq.evaluate(x1x2)

            if self.verbose:
                print("Preparing plots")

            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
            fig.suptitle("Step " + str(self.curr_bo_step), fontsize=16)

            levels = np.linspace(np.min(y_grid), np.max(y_grid), 10)

            # Mean of the GP
            im1 = axes[0, 0].contourf(x1, x2, f_mean.reshape(n_x1, n_x2),
                                      levels=levels)
            axes[0, 0].plot(self.surrogate.X[:, 0], self.surrogate.X[:, 1],
                            'r*',
                            markersize=15)
            if self.curr_bo_step > 0:
                axes[0, 0].plot(p[0][0, 0], p[0][0, 1], 'c*', markersize=20)
            axes[0, 0].plot(self.x_min[0], self.x_min[1], 'y*', markersize=18)
            axes[0, 0].set_title('Mean of the GP')

            # Chosen task
            im2 = axes[0, 1].contourf(x1, x2, y_grid.reshape(n_x1, n_x2),
                                      levels=levels)
            axes[0, 1].plot(self.surrogate.X[:, 0], self.surrogate.X[:, 1],
                            'r*',
                            markersize=15)
            if self.curr_bo_step > 0:
                axes[0, 1].plot(p[0][0, 0], p[0][0, 1], 'c*', markersize=20)
            axes[0, 1].plot(self.x_min[0], self.x_min[1], 'y*', markersize=18)
            axes[0, 1].set_title(text)

            # Stdev of GP
            im3 = axes[1, 0].contourf(x1, x2, f_std.reshape(n_x1, n_x2))
            fig.colorbar(im3, ax=axes[1, 0])
            axes[1, 0].plot(self.surrogate.X[:, 0], self.surrogate.X[:, 1],
                            'r*',
                            markersize=15)
            if self.curr_bo_step > 0:
                axes[1, 0].plot(p[0][0, 0], p[0][0, 1], 'c*', markersize=20)
            axes[1, 0].set_title('Stdev of the GP')

            # Acquisition function
            im4 = axes[1, 1].contourf(x1, x2, acq_dense.reshape(n_x1, n_x2))
            fig.colorbar(im4, ax=axes[1, 1])
            axes[1, 1].plot(self.surrogate.X[:, 0], self.surrogate.X[:, 1],
                            'r*',
                            markersize=15)
            if self.curr_bo_step > 0:
                axes[1, 1].plot(p[0][0, 0], p[0][0, 1], 'c*', markersize=20)
            axes[1, 1].set_title('Acquisition Function')

            # fig.subplots_adjust(right=0.8)
            cax = fig.add_axes([0.91, 0.55, 0.015, 0.35])
            cb = fig.colorbar(im2, cax=cax)
        else:
            raise NotImplementedError

        if not external_call:
            if save_plots:
                self.save_plots_to_disk(fig)
            else:
                plt.show()

        return fig, axes

    def plot_acq(self, acq_func=None, x_batch=None, x_best=None, x_busy=None):

        import pylab as plt
        assert self.surrogate.X.shape[1] == 1, "Debugging acq only in 1D!"

        if x_batch is not None:
            x_batch = np.vstack(x_batch)
        if x_busy is not None:
            x_busy = np.vstack(x_busy)

        x_dense = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 1000)

        if acq_func is not None:
            fig, axes = plt.subplots(nrows=2, ncols=1,
                                     figsize=(12, 9))
            acq_dense = acq_func(x_dense[:, None])
            if x_best is None:
                x_best = np.atleast_2d(x_dense[np.argmax(acq_dense)])

        else:
            fig, axes = plt.subplots(nrows=1, ncols=1,
                                     figsize=(7, 9))
            axes = axes,
            acq_dense = None

        f_mean, f_var = self.surrogate.predict(x_dense[:, None])
        f_var[f_var < 1e-10] = 1e-10
        f_mean = f_mean.flatten()
        f_var = f_var.flatten()
        f_std = np.sqrt(f_var)

        # Mean and variance of surrogate
        axes[0].plot(x_dense, f_mean, 'k', label='Surrogate')
        axes[0].fill_between(x_dense,
                             f_mean + 2 * f_std,
                             f_mean - 2 * f_std, color='b', alpha=0.2)
        axes[0].set_title(f'{acq_func} Surrogate model')

        # Data in the surrogate
        if len(self.surrogate.X) > 0:
            axes[0].plot(self.surrogate.X, self.surrogate.Y_raw, 'b*',
                         markersize=16, label='Data')

        # New point(s)
        axes[0].plot(x_best, self.surrogate.predict(np.atleast_2d(x_best))[0],
                     'r*', markersize=16, label='New')

        if x_batch is not None:
            axes[0].plot(x_batch, self.surrogate.predict(x_batch)[0], 'g*',
                         label="Batch", markersize=16)
            if acq_func is not None:
                axes[1].plot(x_batch, acq_func(x_batch), 'g*',
                             label="Batch", markersize=16)

        if x_busy is not None and len(x_busy) > 0:
            axes[0].plot(x_busy, self.surrogate.predict(x_busy)[0], 'k*',
                         label="Busy", markersize=16)
            if acq_func is not None:
                axes[1].plot(x_busy, acq_func(x_busy), 'k*',
                             label="Busy", markersize=16)

        if acq_func is not None:
            # Acquisition function
            axes[1].plot(x_dense, acq_dense, 'k')

            # New point(s)
            if x_best is not None:
                axes[1].plot(x_best, acq_func(x_best), 'r*',
                             markersize=16, label='New')

        folder = (self.save_plots if isinstance(self.save_plots, str)
                  else './plots/')

        if not os.path.exists(folder):
            os.makedirs(folder)

        if x_batch is None:
            x_batch = []

        fname = os.path.join(folder, f"{self.curr_bo_step}_{len(x_batch)}.pdf")
        fig.savefig(fname)
        if self.verbose:
            print("Saved plot ", fname)
        plt.close()
        fig.show()

    def save_plots_to_disk(self, fig):
        folder = (self.save_plots if isinstance(self.save_plots, str)
                  else './plots/')

        if not os.path.exists(folder):
            os.makedirs(folder)

        prefix = ('' if self.plots_prefix in (None, False)
                  else self.plots_prefix)

        fname = os.path.join(folder, f"{prefix}{self.curr_bo_step}.png")
        fig.savefig(fname)
        if self.verbose:
            print("Saved plot ", fname)
        plt.close()

    def optimize_surrogate_if_needed(self):
        run_opt = False
        if self.optimise_surrogate_model_flag == 'data':
            if len(self.surrogate.Y) >= self.opt_next_at_n_data:
                self.opt_next_at_n_data += self.optimize_every_n_data
                run_opt = True

        elif self.optimise_surrogate_model_flag == 'iter':
            if len(self.curr_bo_step) >= self.opt_next_at_iter:
                self.opt_next_at_iter += self.optimize_every_n_iter
                run_opt = True

        elif self.optimise_surrogate_model_flag == 'always':
            run_opt = True

        if run_opt:
            if self.verbose >= 1:
                print("Optimising surrogate model...")
            self.surrogate.optimize()
            if self.verbose > 1:
                print(
                    f"Surrogate model optimisation complete. "
                    f"New param_array = {self.surrogate.param_array}")

            self.param_array_hist.append(self.surrogate.param_array)
