# -*- coding: utf-8 -*-
#==========================================
# Title:  BatchCoCaBO.py
# Author: Binxin Ru and Ahsan Alvi
# Date:   20 August 2019
# Link:   https://arxiv.org/abs/1906.08878
#==========================================

import math

import GPy
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.bayesopt.batch_bo import BatchBOHeuristic
from utils.bayesopt.executor import JobExecutorInSeriesBlocking
from utils.bayesopt.util import add_hallucinations_to_x_and_y
from methods.CoCaBO_Base import CoCaBO_Base
from utils.ml_utils.models import GP
from utils.ml_utils.models.additive_gp import MixtureViaSumAndProduct, \
    CategoryOverlapKernel, GPWithSomeFixedDimsAtStart


''' Batch CoCaBO algorithm '''
class BatchCoCaBO(CoCaBO_Base):

    def __init__(self, objfn, initN, bounds, acq_type, C, **kwargs):

        super(BatchCoCaBO, self).__init__(objfn, initN, bounds, acq_type, C,
                                          **kwargs)
        self.best_val_list = []
        self.C_list = self.C
        self.name = 'BCoCaBO'

    def runOptim(self, budget, seed, initData=None,
                 initResult=None, ):

        if (initData and initResult):
            self.data = initData[:]
            self.result = initResult[:]
        else:
            self.data, self.result = self.initialise(seed)

        bestUpperBoundEstimate = 2 * budget / 3

        gamma_list = [np.sqrt(C * math.log(C / self.batch_size) / (
                (math.e - 1) * self.batch_size * bestUpperBoundEstimate))
                      for C in self.C_list]
        gamma_list = [g if not np.isnan(g) else 1 for g in gamma_list]

        Wc_list_init = [np.ones(C) for C in self.C_list]
        Wc_list = Wc_list_init
        nDim = len(self.bounds)

        result_list = []
        starting_best = np.max(-1 * self.result[0])
        result_list.append([-1, None, None, starting_best, None])

        continuous_dims = list(range(len(self.C_list), nDim))
        categorical_dims = list(range(len(self.C_list)))

        for t in tqdm(range(budget)):
            self.iteration = t
            ht_batch_list, probabilityDistribution_list, S0 = self.compute_prob_dist_and_draw_hts(
                Wc_list, gamma_list, self.batch_size)

            ht_batch_list = ht_batch_list.astype(int)

            # Obtain the reward for multi-armed bandit: B x len(self.C_list)
            Gt_ht_list = self.RewardperCategoryviaBO(self.f, ht_batch_list,
                                                     categorical_dims,
                                                     continuous_dims)

            # Update the reward and the weight
            Wc_list = self.update_weights_for_all_cat_var(Gt_ht_list,
                                                          ht_batch_list,
                                                          Wc_list, gamma_list,
                                                          probabilityDistribution_list,
                                                          self.batch_size,
                                                          S0=S0)

            # Get the best value till now
            besty, li, vi = self.getBestVal2(self.result)

            # Store the results of this iteration
            result_list.append(
                [t, ht_batch_list, Gt_ht_list, besty, self.mix_used,
                 self.model_hp])
            self.ht_recommedations.append(ht_batch_list)

        df = pd.DataFrame(result_list,
                          columns=["iter", "ht", "Reward", "best_value",
                                   "mix_val", "model_hp"])
        bestx = self.data[li][vi]
        self.best_val_list.append(
            [self.batch_size, self.trial_num, li, besty, bestx])
        return df

    # =============================================================================
    #   Function returns the reward for multi-armed bandit
    # =============================================================================
    def RewardperCategoryviaBO(self, objfn, ht_next_batch_list,
                               categorical_dims,
                               continuous_dims):

        #  Get observation data
        Zt = self.data[0]
        yt = self.result[0]

        my_kernel, hp_bounds = self.get_kernel(categorical_dims,
                                               continuous_dims)

        gp_opt_params = {'method': 'multigrad',
                         'num_restarts': 5,
                         'restart_bounds': hp_bounds,
                         'hp_bounds': hp_bounds,
                         'verbose': False}

        gp_kwargs = {'y_norm': 'meanstd',
                     'opt_params': gp_opt_params}
        gp_args = (Zt, yt, my_kernel)

        gp = GP(*gp_args, **gp_kwargs)

        opt_flag, gp = self.set_model_params_and_opt_flag(gp)
        if opt_flag:
            # print("\noptimising!\n")
            gp.optimize()
        self.model_hp = gp.param_array

        acq_dict = {'type': 'subspace'}

        acq_opt_params = {'method': 'samplegrad',
                          'num_local': 5,
                          'num_samples': 5000,
                          'num_chunks': 10,
                          'verbose': False}

        ymin_opt_params = {'method': 'standard'}

        # Find the unique combinations in h and their frequency
        h_unique, h_counts = np.unique(ht_next_batch_list,
                                       return_counts=True, axis=0)

        # Create the batch
        z_batch_list = []
        for idx, curr_h in enumerate(h_unique):
            # Perform batch BO with a single fixed h
            gp_for_bo = GPWithSomeFixedDimsAtStart(*gp_args,
                                                   fixed_dim_vals=curr_h,
                                                   **gp_kwargs)
            gp_for_bo.param_array = gp.param_array

            curr_batch_size = h_counts[idx]
            interface = JobExecutorInSeriesBlocking(curr_batch_size)

            # Adding repulsion effect to already-selected locations
            if len(z_batch_list) > 0:
                self.surrogate = gp_for_bo
                self.async_infill_strategy = 'kriging_believer'  # hack
                surrogate_x_with_fake, surrogate_y_with_fake = \
                    add_hallucinations_to_x_and_y(
                        self, gp_for_bo.X, gp_for_bo.Y_raw,
                        np.vstack(z_batch_list))
                gp_for_bo.set_XY(X=surrogate_x_with_fake,
                                 Y=surrogate_y_with_fake)

            bo = BatchBOHeuristic(objfn, gp_for_bo, self.x_bounds,
                                  async_infill_strategy='kriging_believer',
                                  offset_acq=True,
                                  async_interface=interface,
                                  batch_size=curr_batch_size,
                                  acq_dict=acq_dict,
                                  y_min_opt_params=ymin_opt_params,
                                  acq_opt_params=acq_opt_params,
                                  optimise_surrogate_model=False)

            x_batch_for_curr_h, _ = bo.get_next()

            z_batch_for_curr_h = np.hstack((
                np.vstack([curr_h] * curr_batch_size),
                np.vstack(x_batch_for_curr_h)
            ))
            z_batch_list.append(z_batch_for_curr_h)

        z_batch_next = np.vstack(z_batch_list)

        #  Evaluate objective function at
        y_batch_next = np.zeros((self.batch_size, 1))
        for b in range(self.batch_size):
            x_next = z_batch_next[b, continuous_dims]
            ht_next_list = z_batch_next[b, categorical_dims]
            try:
                y_next = objfn(ht_next_list, x_next)
            except:
                print('stop')

            y_batch_next[b] = y_next

        # Append recommeded data
        self.mix_used = gp.kern.mix[0]
        self.data[0] = np.row_stack((self.data[0], z_batch_next))
        self.result[0] = np.row_stack((self.result[0], y_batch_next))

        # Obtain the reward for each categorical variable: B x len(self.C_list)
        ht_batch_list_rewards = self.compute_reward_for_all_cat_variable(
            ht_next_batch_list, self.batch_size)

        bestval_ht = np.max(self.result[0] * -1)
        # print(f'arm pulled={ht_next_batch_list[:]} ; '
        #       f'\n rewards = {ht_batch_list_rewards[:]}; '
        #       f'y_best = {bestval_ht}; mix={self.mix_used}')
        print(f'arm pulled={ht_next_batch_list[:]} ; '
              f'y_best = {bestval_ht}; mix={self.mix_used}')

        return ht_batch_list_rewards

    def get_kernel(self, categorical_dims, continuous_dims):
        # Create surrogate model
        if self.ARD:
            hp_bounds = np.array([
                *[[1e-4, 3]] * len(continuous_dims),  # cont lengthscale
                [1e-6, 1],  # likelihood variance
            ])
        else:
            hp_bounds = np.array([
                [1e-4, 3],  # cont lengthscale
                [1e-6, 1],  # likelihood variance
            ])
        fix_mix_in_this_iter, mix_value, hp_bounds = self.get_mix(hp_bounds)
        k_cat = CategoryOverlapKernel(len(categorical_dims),
                                      active_dims=categorical_dims)  # cat
        k_cont = GPy.kern.Matern52(len(continuous_dims),
                                   lengthscale=self.default_cont_lengthscale,
                                   active_dims=continuous_dims,
                                   ARD=self.ARD)  # cont
        my_kernel = MixtureViaSumAndProduct(
            len(categorical_dims) + len(continuous_dims),
            k_cat, k_cont, mix=mix_value, fix_inner_variances=True,
            fix_mix=fix_mix_in_this_iter)
        return my_kernel, hp_bounds
