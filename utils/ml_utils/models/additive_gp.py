# -*- coding: utf-8 -*-
#==========================================
# Title:  additive_gp.py
# Author: Binxin Ru and Ahsan Alvi
# Date:   20 August 2019
# Link:   https://arxiv.org/abs/1906.08878
#==========================================

from typing import Union, Tuple

import GPy
import numpy as np
from paramz.transformations import Logexp

from utils.ml_utils.models import GP


class GPWithSomeFixedDimsAtStart(GP):
    """
    Utility class that allows for predict() interface while only providing
    a subset of the inputs and filling in the missing ones.

    If the fixed dims are h and the provided values are x,
    then the predict() function returns the posterior at z = [h, x]
    """

    def __init__(self, *args, fixed_dim_vals=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert fixed_dim_vals is not None
        self.fixed_dim_vals = np.array(fixed_dim_vals).flatten()

    def add_fixed_to_x(self, x_star):
        h_star = np.vstack([self.fixed_dim_vals] * len(x_star))
        z_star = np.hstack((h_star, x_star))
        return z_star

    def predict_latent(self, x_star: np.ndarray, full_cov: bool = False,
                       kern=None):
        """
        Predict at z = [h, x]
        """

        return super().predict_latent(self.add_fixed_to_x(x_star),
                                      full_cov, kern)

    def dposterior_dx(self, x_star: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        return super().dposterior_dx(self.add_fixed_to_x(x_star))


class MixtureViaSumAndProduct(GPy.kern.Kern):
    """
    Kernel of the form

    k = (1-mix)*(k1 + k2) + mix*k1*k2


    Parameters
    ----------
    input_dim
        number of all dims (for k1 and k2 together)
    k1
        First kernel
    k2
        Second kernel
    active_dims
        active dims of this kernel
    mix
        see equation above
    fix_variances
        unlinks the variance parameters if set to True
    fix_mix
        Does not register mix as a parameter that can be learned

    """

    def __init__(self, input_dim: int, k1: GPy.kern.Kern, k2: GPy.kern.Kern,
                 active_dims: Union[list, np.ndarray] = None, variance=1.0,
                 mix: float = 0.5,
                 fix_inner_variances: bool = False, fix_mix=True,
                 fix_variance=True):

        super().__init__(input_dim, active_dims, 'MixtureViaSumAndProduct')

        self.acceptable_kernels = (GPy.kern.RBF, GPy.kern.Matern52,
                                   CategoryOverlapKernel
                                   )

        assert isinstance(k1, self.acceptable_kernels)
        assert isinstance(k2, self.acceptable_kernels)

        self.mix = GPy.core.parameterization.Param('mix', mix, Logexp())
        self.variance = GPy.core.parameterization.Param('variance', variance,
                                                        Logexp())

        self.fix_variance = fix_variance
        if not self.fix_variance:
            self.link_parameter(self.variance)

        # If we are learning the mix, then add it as a visible param
        self.fix_mix = fix_mix
        if not self.fix_mix:
            self.link_parameter(self.mix)

        self.k1 = k1
        self.k2 = k2

        self.fix_inner_variances = fix_inner_variances
        if self.fix_inner_variances:
            self.k1.unlink_parameter(self.k1.variance)
            self.k2.unlink_parameter(self.k2.variance)

        self.link_parameters(self.k1, self.k2)

    def get_dk_dtheta(self, k: GPy.kern.Kern, X, X2=None):
        assert isinstance(k, self.acceptable_kernels)

        if X2 is None:
            X2 = X
        X_sliced, X2_sliced = X[:, k.active_dims], X2[:, k.active_dims]

        if isinstance(k, (GPy.kern.RBF, GPy.kern.Matern52)):
            dk_dr = k.dK_dr_via_X(X_sliced, X2_sliced)

            # dr/dl
            if k.ARD:
                tmp = k._inv_dist(X_sliced, X2_sliced)
                dr_dl = -np.dstack([tmp * np.square(
                    X_sliced[:, q:q + 1] - X2_sliced[:, q:q + 1].T) /
                                    k.lengthscale[q] ** 3
                                    for q in range(k.input_dim)])
                dk_dl = dk_dr[..., None] * dr_dl
            else:
                r = k._scaled_dist(X_sliced, X2_sliced)
                dr_dl = - r / k.lengthscale
                dk_dl = dk_dr * dr_dl

            # # For testing the broadcast multiplication
            # dk_dl_slow = []
            # for ii in range(dr_dl.shape[-1]):
            #     dr_dlj = dr_dl[...,ii]
            #     dk_dlj = dk_dr * dr_dlj
            #     dk_dl_slow.append(dk_dlj)
            #
            # dk_dl_slow = np.dstack(dk_dl_slow)

        elif isinstance(k, CategoryOverlapKernel):
            dk_dl = None

        else:
            raise NotImplementedError

        # Return variance grad as well, if not fixed
        if not self.fix_inner_variances:
            return k.K(X, X2) / k.variance, dk_dl
        else:
            return dk_dl

    def update_gradients_full(self, dL_dK, X, X2=None):

        # This gets the values of dk/dtheta as a NxN matrix (no summations)
        if X2 is None:
            X2 = X
        dk1_dtheta1 = self.get_dk_dtheta(self.k1, X, X2)  # N x N
        dk2_dtheta2 = self.get_dk_dtheta(self.k2, X, X2)  # N x N

        # Separate the variance and lengthscale grads (for ARD purposes)
        if self.fix_inner_variances:
            dk1_dl1 = dk1_dtheta1
            dk2_dl2 = dk2_dtheta2
            dk1_dvar1 = []
            dk2_dvar2 = []
        else:
            dk1_dvar1, dk1_dl1 = dk1_dtheta1
            dk2_dvar2, dk2_dl2 = dk2_dtheta2

        # Evaluate each kernel over its own subspace
        k1_xx = self.k1.K(X, X2)  # N x N
        k2_xx = self.k2.K(X, X2)  # N x N

        # dk/dl for l1 and l2
        # If gradient is None, then vars other than lengthscale don't exist.
        # This is relevant for the CategoryOverlapKernel
        if dk1_dl1 is not None:
            # ARD requires a summation along last axis for each lengthscale
            if hasattr(self.k1, 'ARD') and self.k1.ARD:
                dk_dl1 = np.sum(
                    dL_dK[..., None] * (
                            0.5 * dk1_dl1 * (1 - self.mix) * self.variance
                            + self.mix * self.variance * dk1_dl1 *
                            k2_xx[..., None]),
                    (0, 1))
            else:
                dk_dl1 = np.sum(
                    dL_dK * (0.5 * dk1_dl1 * (1 - self.mix) * self.variance
                             + self.mix * self.variance * dk1_dl1 * k2_xx))
        else:
            dk_dl1 = []

        if dk2_dl2 is not None:
            if hasattr(self.k2, 'ARD') and self.k2.ARD:
                dk_dl2 = np.sum(
                    dL_dK[..., None] * (
                            0.5 * dk2_dl2 * (1 - self.mix) * self.variance
                            + self.mix * self.variance * dk2_dl2 *
                            k1_xx[..., None]),
                    (0, 1))
            else:
                dk_dl2 = np.sum(
                    dL_dK * (0.5 * dk2_dl2 * (1 - self.mix) * self.variance
                             + self.mix * self.variance * dk2_dl2 * k1_xx))
        else:
            dk_dl2 = []

        # dk/dvar for var1 and var 2
        if self.fix_inner_variances:
            dk_dvar1 = []
            dk_dvar2 = []
        else:
            dk_dvar1 = np.sum(
                dL_dK * (0.5 * dk1_dvar1 * (1 - self.mix) * self.variance
                         + self.mix * self.variance * dk1_dvar1 * k2_xx))
            dk_dvar2 = np.sum(
                dL_dK * (0.5 * dk2_dvar2 * (1 - self.mix) * self.variance
                         + self.mix * self.variance * dk2_dvar2 * k1_xx))

        # Combining the gradients into one vector and updating
        dk_dtheta1 = np.hstack((dk_dvar1, dk_dl1))
        dk_dtheta2 = np.hstack((dk_dvar2, dk_dl2))
        self.k1.gradient = dk_dtheta1
        self.k2.gradient = dk_dtheta2

        # if not self.fix_mix:
        self.mix.gradient = np.sum(dL_dK *
                                   (-0.5 * (k1_xx + k2_xx) +
                                    (k1_xx * k2_xx))) * self.variance

        # if not self.fix_variance:
        self.variance.gradient = \
            np.sum(self.K(X, X2) * dL_dK) / self.variance

    def K(self, X, X2=None):
        k1_xx = self.k1.K(X, X2)
        k2_xx = self.k2.K(X, X2)
        return self.variance * ((1 - self.mix) * 0.5 * (k1_xx + k2_xx)
                                + self.mix * k1_xx * k2_xx)

    def gradients_X(self, dL_dK, X, X2, which_k=2):
        """
        This function evaluates the gradients w.r.t. the kernel's inputs.
        Default is set to the second kernel, due to this function's
        use in categorical+continuous BO requiring gradients w.r.t.
        the continuous space, which is generally the second kernel.

        which_k = 1  # derivative w.r.t. k1 space
        which_k = 2  # derivative w.r.t. k2 space
        """
        active_kern, other_kern = self.get_active_kernel(which_k)

        # Evaluate the kernel grads in a loop, as the function internally
        # sums up results, which is something we want to avoid until
        # the last step
        active_kern_grads = np.zeros((len(X), len(X2), self.input_dim))
        for ii in range(len(X)):
            for jj in range(len(X2)):
                active_kern_grads[ii, jj, :] = \
                    active_kern.gradients_X(
                        np.atleast_2d(dL_dK[ii, jj]),
                        np.atleast_2d(X[ii]),
                        np.atleast_2d(X2[jj]))

        other_kern_vals = other_kern.K(X, X2)

        out = np.sum(active_kern_grads *
                     (1 - self.mix + self.mix * other_kern_vals[..., None]),
                     axis=1)
        return out

    def gradients_X_diag(self, dL_dKdiag, X, which_k=2):
        active_kern, other_kern = self.get_active_kernel(which_k)
        if isinstance(active_kern, GPy.kern.src.stationary.Stationary):
            return np.zeros(X.shape)
        else:
            raise NotImplementedError("gradients_X_diag not implemented "
                                      "for this type of kernel")

    def get_active_kernel(self, which_k):
        if which_k == 1:
            active_kern = self.k1
            other_kern = self.k2
        elif which_k == 2:
            active_kern = self.k2
            other_kern = self.k1
        else:
            raise NotImplementedError(f"Bad selection of which_k = {which_k}")
        return active_kern, other_kern


class CategoryOverlapKernel(GPy.kern.Kern):
    """
    Kernel that counts the number of categories that are the same
    between inputs and returns the normalised similarity score:

    k = variance * 1/N_c * (degree of overlap)
    """

    def __init__(self, input_dim, variance=1.0, active_dims=None,
                 name='catoverlap'):
        super().__init__(input_dim, active_dims=active_dims, name=name)
        self.variance = GPy.core.parameterization.Param('variance',
                                                        variance, Logexp())
        self.link_parameter(self.variance)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        # Counting the number of categories that are the same using GPy's
        # broadcasting approach
        diff = X[:, None] - X2[None, :]
        # nonzero location = different cat
        diff[np.where(np.abs(diff))] = 1
        # invert, to now count same cats
        diff1 = np.logical_not(diff)
        # dividing by number of cat variables to keep this term in range [0,1]
        k_cat = self.variance * np.sum(diff1, -1) / self.input_dim
        return k_cat

    def update_gradients_full(self, dL_dK, X, X2=None):
        self.variance.gradient = np.sum(self.K(X, X2) * dL_dK) / self.variance
