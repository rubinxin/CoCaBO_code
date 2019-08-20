# -*- coding: utf-8 -*-
#==========================================
# Title:  async_bo.py
# Author: Binxin Ru and Ahsan Alvi
# Date:   20 August 2019
# Link:   https://arxiv.org/abs/1906.08878
#==========================================

"""Async Bayesian optimization classes"""
import sys
import time
from typing import Optional, Callable

import numpy as np
import pandas as pd

from utils.bayesopt.util import add_hallucinations_to_x_and_y
from utils.ml_utils import timed_print as print
from utils.ml_utils.models import GP
from .bayesopt import BayesianOptimisation
from .executor import ExecutorBase


class AsyncBayesianOptimization(BayesianOptimisation):
    """Async Bayesian optimization class

    Performs Bayesian optimization with a set number of busy and free workers

    Parameters
    ----------
    sampler : Callable
        function handle returning sample from expensive function being
        optimized

    surrogate : basic_gp.GP
        (GP) model that models the surface of 'objective'

    bounds : ndarray
        bounds of each dimension of x as a Dx2 vector (default [0, 1])

    async_interface : ExecutorBase
        Interface that deals with exchange of information between
        async workers and the BO loop

    batch_size : int
        How many tasks to suggest in one go. This will wait for the
        required number of workers to become free before evaluating the batch

    acq_dict : acquisition.AcquisitionFunction
        Defaults to EI

    starting_jobs : list(dicts)
        list of dicts in the form {'x': np.ndarray, 'f': callable, 't': float}

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
                 async_interface: ExecutorBase = None,
                 starting_jobs: Optional[list] = None,
                 **kwargs):

        self.starting_jobs = starting_jobs
        self.interface = async_interface

        super().__init__(sampler, surrogate, bounds,
                         **kwargs)

    def _initialise_bo_df(self):
        """
        Initialise the DataFrame for keeping track of the BO run
        """
        self.df = pd.DataFrame(
            columns=['ii', 't', 'y_min', 'x_min', 'n_busy', 'x_busy', 'n_data',
                     'model_x', 'model_y', 'model_param_array',
                     'acq_at_sample', 'requested_x_sample', 'x_sample',
                     'y_sample', 'time_taken_opt_surrogate',
                     'time_taken_find_y_min', 'time_taken_get_next',
                     'time_taken_bo_step', 'var_at_y_min', 'cond_k'])

        self.x_min, self.y_min, self.var_at_y_min = self._get_y_min()
        if self.starting_jobs is not None:
            x_busy = np.vstack([job['x']
                                for job in
                                self.starting_jobs])
        else:
            x_busy = None

        starting_record = {'ii': -1,
                           'iteration': 0,
                           't': self.interface.status['t'],
                           'y_min': self.y_min,
                           'x_min': self.x_min,
                           'n_busy': self.interface.n_busy_workers,
                           'x_busy': x_busy,
                           'n_free': self.interface.n_free_workers,
                           'n_data': len(self.surrogate.X),
                           'model_x': self.surrogate.X,
                           'model_y': self.surrogate.Y,
                           'model_param_array': self.surrogate.param_array,
                           'acq_at_sample': np.nan,
                           'requested_x_sample': np.nan,
                           'y_sample': np.nan,
                           'x_sample': np.nan,
                           'time_taken_opt_surrogate': np.nan,
                           'time_taken_find_y_min': np.nan,
                           'time_taken_get_next': np.nan,
                           'time_taken_bo_step': np.nan,
                           'var_at_y_min': self.var_at_y_min,
                           'cond_k': np.nan}
        self.df = self.df.append([starting_record], sort=True)

    def _update_bo_df(self, x_batch, acq_at_x_best, new_sample_x, new_sample_y,
                      time_dict):
        """Updates the local dataframe with the current iteration's data

        Parameters
        ----------
        x_batch
            Best location to sample at
        acq_at_x_best
            Acquisition function value at x_best
        new_sample_x
            actual sample received
        new_sample_y
            actual sample received
        time_dict
            time taken for different parts of the algo in seconds

        """

        # requested_x_sample = new points queued in the async worker
        current_record = {
            'ii': self.curr_bo_step,
            't': self.interface.status['t'],
            'iteration': self.curr_bo_step + 1,
            'y_min': self.y_min,
            'x_min': self.x_min,
            'n_busy': self.interface.n_busy_workers,
            'x_busy': self.interface.get_array_of_running_jobs(),
            'n_free': self.interface.n_free_workers,
            'n_data': len(self.surrogate.X),
            'model_x': self.surrogate.X,
            'model_y': self.surrogate.Y,
            'model_param_array': self.surrogate.param_array,
            'acq_at_sample': acq_at_x_best,
            'requested_x_sample': x_batch,
            'y_sample': new_sample_y,
            'x_sample': new_sample_x,
            'time_taken_opt_surrogate': time_dict['time_taken_opt_surrogate'],
            'time_taken_find_y_min': time_dict['time_taken_find_y_min'],
            'time_taken_get_next': time_dict['time_taken_get_next'],
            'time_taken_bo_step': time_dict['time_taken_bo_step'],
            'var_at_y_min': self.var_at_y_min,
            'cond_k': (self.cond_k_hist[
                           self.curr_bo_step] if self.track_cond_k else None)
        }
        self.df = self.df.append([current_record], sort=True)

    def run(self):
        """
        Run the Async BayesOpt loop
        """
        t_starting_run = time.time()
        if self.verbose:
            print("Started BayesOpt.run()")

        self._initialise_bo_df()

        if self.starting_jobs is not None:
            for job in self.starting_jobs:
                self.interface.add_job_to_queue(job)

        for self.curr_bo_step in range(0, self.n_bo_steps):
            new_sample_x, new_sample_y = None, None
            # try:
            if True:
                t_beginning_of_bo_step = time.time()
                if self.verbose:
                    print("**--** Starting BayesOpt iteration {}/{} **--**"
                          .format(self.curr_bo_step + 1, self.n_bo_steps))

                # Move time ahead until we have the correct number of free
                # workers
                self.interface.run_until_n_free(self.batch_size)
                n_free_workers = self.interface.status['n_free_workers']

                completed_jobs = self.interface.get_completed_jobs()
                if len(completed_jobs) > 0:
                    new_sample_x, new_sample_y = \
                        self._add_completed_jobs_to_surrogate(completed_jobs)
                assert n_free_workers >= self.batch_size

                t_before_opt_surrogate = time.time()
                # if self.verbose:
                #     print(f"Surrogate n_data = {len(self.surrogate.X)}")
                # if self.optimise_surrogate_model_flag:
                #     if self.verbose > 1:
                #         print("Optimising surrogate model...")
                #     self.surrogate.optimize()
                #     self.param_array_hist.append(self.surrogate.param_array)
                #     if self.verbose > 1:
                #         print(
                #             f"Surrogate model optimisation complete. "
                #           f"New param_array = {self.surrogate.param_array}")
                self.optimize_surrogate_if_needed()

                t_after_opt_surrogate = time.time()

                t_before_find_y_min = time.time()
                self.x_min, self.y_min, self.var_at_y_min = self._get_y_min()
                t_after_find_y_min = t_before_get_next = time.time()
                if self.verbose:
                    print("Selecting next point(s)...")
                x_batch, acq_at_x_batch = self.get_next()
                t_after_get_next = t_end_of_bo_step = time.time()

                time_taken_opt_surrogate = \
                    (t_after_opt_surrogate - t_before_opt_surrogate)
                time_taken_find_y_min = \
                    (t_after_find_y_min - t_before_find_y_min)
                time_taken_get_next = \
                    (t_after_get_next - t_before_get_next)
                time_taken_bo_step = \
                    (t_end_of_bo_step - t_beginning_of_bo_step)

                time_taken_dict = {
                    'time_taken_opt_surrogate': time_taken_opt_surrogate,
                    'time_taken_find_y_min': time_taken_find_y_min,
                    'time_taken_get_next': time_taken_get_next,
                    'time_taken_bo_step': time_taken_bo_step, }

                if self.create_plots:
                    self.plot_step(x_batch=x_batch)

                # queue the jobs
                jobs = []
                for ii in range(len(x_batch)):
                    job = {'x': x_batch[ii], 'f': self.sampler}
                    jobs.append(job)

                self.interface.add_job_to_queue(jobs)

                self.save_history(None)

                if self.curr_bo_step == self.n_bo_steps - 1:  # last step
                    if self.verbose > 1:
                        print("Used up budget.")
                        print("Minimum at",
                              self.surrogate.X[np.argmin(self.surrogate.Y)])

                self._update_bo_df(x_batch, acq_at_x_batch, new_sample_x,
                                   new_sample_y, time_taken_dict)

                # Attempting to force SLURM to update the output file
                sys.stdout.flush()

            # except np.linalg.linalg.LinAlgError:
            #     print("WARNING: BayesOpt crashed at iteration {}!".format(
            #         self.curr_bo_step))
            #     break
        if self.verbose:
            print(
                f"Completed BO exp in;"
                f" {round(time.time() - t_starting_run, 2)}s")

    def get_next(self):
        """Finds the next point(s) to sample at

        Returns
        -------
        x_best : np.ndarray
            Location to sample at
        acq_at_x_best : float
            Value of the acquisition function at the sampling locations
        """
        raise NotImplementedError

    def _add_completed_jobs_to_surrogate(self, completed_jobs):
        x = []
        y = []
        for job in completed_jobs:
            x.append(job['x'])
            y.append(job['y'])
        x = np.vstack(x)
        y = np.vstack(y)
        self._update_surrogate_with_new_data(x, y)
        return x, y

    def plot_step(self, x_batch=None, save_plots=None, **kwargs):

        if save_plots is None:
            save_plots = self.save_plots

        if isinstance(x_batch, list):
            x_batch = np.vstack(x_batch)

        fig, axes = super().plot_step(x_batch, external_call=True)

        acq = self._create_acq_function()

        if len(self.bounds) == 1:  # 1D
            x_busy = self.interface.get_array_of_running_jobs()
            if x_busy is not None:
                axes[0].plot(x_busy, self.surrogate.predict(x_busy)[0], 'g*',
                             label="Busy", markersize=16)
                axes[1].plot(x_busy, acq.evaluate(x_busy), 'g*',
                             label="Busy", markersize=16)

            axes[0].legend(numpoints=1)
            axes[1].legend(numpoints=1)

        if save_plots:
            self.save_plots_to_disk(fig)
        else:
            fig.show()

        return fig, axes


class AsyncBOHeuristicQEI(AsyncBayesianOptimization):
    """Async BO with approximate q-EI

    Q-EI is approximated by sequentially finding the best location and
    setting its y-value using one of Ginsbourger's heuristics until the
    batch is full
    """

    def __init__(self, sampler, surrogate, bounds,
                 async_infill_strategy='kriging_believer',
                 **kwargs):
        from utils.ml_utils.models.additive_gp import GPWithSomeFixedDimsAtStart

        if async_infill_strategy is None:
            self.async_infill_strategy = 'constant_liar_min'
        else:
            self.async_infill_strategy = async_infill_strategy

        if isinstance(surrogate, GPWithSomeFixedDimsAtStart):
            self.mabbo = True
        elif isinstance(surrogate, GP):
            self.mabbo = False
        else:
            raise NotImplementedError

        super().__init__(sampler, surrogate, bounds, **kwargs)

    def get_next(self):
        """Finds the next point(s) to sample at

        This function interacts with the async interface to get info about
        completed and running jobs and computes the next point(s) to add
        to the queue based on the batch size

        Returns
        -------
        x_best : np.ndarray
            Location to sample at
        acq_at_x_best : float
            Value of the acquisition function at the sampling locations
        """
        old_surrogate_x = self.surrogate.X
        old_surrogate_y = self.surrogate.Y_raw
        x_busy = self.interface.get_array_of_running_jobs()

        if self.mabbo:
            fixed_dim_vals = self.surrogate.fixed_dim_vals
        else:
            fixed_dim_vals = None

        surrogate_x_with_fake, surrogate_y_with_fake = \
            add_hallucinations_to_x_and_y(self, old_surrogate_x,
                                          old_surrogate_y, x_busy,
                                          fixed_dim_vals=fixed_dim_vals)
        self.surrogate.set_XY(X=surrogate_x_with_fake,
                              Y=surrogate_y_with_fake)

        acq = self._create_acq_function()
        x_best, acq_at_x_best = self._optimise_acq_func(acq)

        x_batch = [x_best, ]
        acq_at_each_x_batch = [acq_at_x_best, ]
        if self.batch_size > 1:
            for ii in range(self.batch_size - 1):
                # Using the async infill heuristic
                current_surrogate_x = self.surrogate.X
                current_surrogate_y = self.surrogate.Y_raw

                surrogate_x_with_fake, surrogate_y_with_fake = \
                    add_hallucinations_to_x_and_y(
                        self, current_surrogate_x,
                        current_surrogate_y, x_batch,
                        fixed_dim_vals=fixed_dim_vals)

                self.surrogate.set_XY(X=surrogate_x_with_fake,
                                      Y=surrogate_y_with_fake)

                acq = self._create_acq_function()
                x_best, acq_at_x_best = self._optimise_acq_func(acq)

                x_batch.append(x_best)
                acq_at_each_x_batch.append(acq_at_x_best)

        self.surrogate.set_XY(X=old_surrogate_x, Y=old_surrogate_y)

        assert len(x_batch) == self.batch_size

        return x_batch, acq_at_each_x_batch
