# -*- coding: utf-8 -*-

import math

import GPy
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.bayesopt.acquisition import AcquisitionOnSubspace, EI, UCB
from methods.CoCaBO_Base import CoCaBO_Base
from utils.ml_utils.models import GP
from utils.ml_utils.models.additive_gp import MixtureViaSumAndProduct, \
    CategoryOverlapKernel
from utils.ml_utils.optimization import sample_then_minimize

''' Sequential CoCaBO algorithm '''
class CoCaBO(CoCaBO_Base):

    def __init__(self, objfn, initN, bounds, acq_type, C, **kwargs):

        super(CoCaBO, self).__init__(objfn, initN, bounds, acq_type, C, **kwargs)
        self.best_val_list = []
        self.C_list = self.C
        self.name = 'CoCaBO'

    def runOptim(self, budget, seed, batch_size=1, initData=None, initResult=None):

        if (initData and initResult):
            self.data = initData[:]
            self.result = initResult[:]
        else:
            self.data, self.result = self.initialise(seed)

        # Initialize wts and probs
        b = batch_size
        bestUpperBoundEstimate = 2 * budget / 3
        gamma_list = [math.sqrt(C * math.log(C) /
                                ((math.e - 1) * bestUpperBoundEstimate))
                      for C in self.C_list]
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

            # Compute the probability for each category and Choose categorical variables
            ht_list, probabilityDistribution_list = \
                self.compute_prob_dist_and_draw_hts(Wc_list, gamma_list,
                                                    batch_size)

            # Get reward for multi-armed bandit
            Gt_ht_list = self.RewardperCategoryviaBO(self.f, ht_list,
                                                     categorical_dims,
                                                     continuous_dims,
                                                     self.bounds,
                                                     self.acq_type, b)

            # Update the reward and the weight
            Wc_list = self.update_weights_for_all_cat_var(
                Gt_ht_list, ht_list,
                Wc_list, gamma_list,
                probabilityDistribution_list,
                batch_size)

            # Get the best value till now
            besty, li, vi = self.getBestVal2(self.result)

            # Store the results of this iteration
            result_list.append([t, ht_list, Gt_ht_list, besty, self.mix_used,
                                self.model_hp])

            self.ht_recommedations.append(ht_list)

        df = pd.DataFrame(result_list, columns=["iter", "ht", "Reward",
                                                "best_value", "mix_val",
                                                "model_hp"])
        bestx = self.data[li][vi]
        self.best_val_list.append([batch_size, self.trial_num, li, besty,
                                   bestx])

        return df

    # =============================================================================
    #   Function returns the reward for multi-armed bandit
    # =============================================================================
    def RewardperCategoryviaBO(self, objfn, ht_next_list, categorical_dims,
                               continuous_dims, bounds, acq_type, b):

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

        gp = GP(Zt, yt, my_kernel, y_norm='meanstd',
                opt_params=gp_opt_params)

        opt_flag, gp = self.set_model_params_and_opt_flag(gp)
        if opt_flag:
            print("\noptimising!\n")
            gp.optimize()
        self.model_hp = gp.param_array


        self.mix_used = gp.kern.mix[0]

        x_bounds = np.array([d['domain'] for d in bounds
                             if d['type'] == 'continuous'])
        # create acq
        if acq_type == 'EI':
            acq = EI(gp, np.min(gp.Y_raw))
        elif acq_type == 'LCB':
            acq = UCB(gp, 2.0)

        acq_sub = AcquisitionOnSubspace(acq, my_kernel.k2.active_dims,
                                        ht_next_list)

        def optimiser_func(x):
            return -acq_sub.evaluate(np.atleast_2d(x))

        res = sample_then_minimize(
            optimiser_func,
            x_bounds,
            num_samples=5000,
            num_chunks=10,
            num_local=3,
            minimize_options=None,
            evaluate_sequentially=False)

        x_next = res.x
        z_next = np.hstack((ht_next_list, x_next))

        #  Evaluate objective function at z_next = [x_next,  ht_next_list]
        y_next = objfn(ht_next_list, x_next)

        # Append recommeded data
        self.data[0] = np.row_stack((self.data[0], z_next))
        self.result[0] = np.row_stack((self.result[0], y_next))

        # Obtain the reward for each categorical variable
        ht_next_list_array = np.atleast_2d(ht_next_list)
        ht_list_rewards = self.compute_reward_for_all_cat_variable(
            ht_next_list_array, b)
        ht_list_rewards = list(ht_list_rewards.flatten())

        bestval_ht = np.max(self.result[0] * -1)
        print(f'arm pulled={ht_next_list[:]} ; rewards = {ht_list_rewards[:]};'
              f' y_best = {bestval_ht}; mix={self.mix_used}')

        return ht_list_rewards

    def get_kernel(self, categorical_dims, continuous_dims):
        # create surrogate
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
