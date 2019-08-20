# -*- coding: utf-8 -*-
#==========================================
# Title:  gp.py
# Author: Binxin Ru and Ahsan Alvi
# Date:   20 August 2019
# Link:   https://arxiv.org/abs/1906.08878
#==========================================

"""
This module contains the basic Gaussian process class.

This class is used as a standard GP model but also as a base for the
specific noisy-input GP models in nigp.py and randfuncgp.py.
"""

from pprint import pprint
from typing import Union, Tuple, List, Optional

import GPy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.optimize import OptimizeResult

from ..misc import timed_print as print
from ..optimization import minimize_with_restarts, sample_then_minimize


class GP(object):
    """Gaussian Process model

    Using GPy model structure as inspiration.

    Parameters
    ----------
    X
        input observations

    Y
        observed values

    kern
        a GPy kernel, defaults to rbf

    hyper_priors
        list of GPy.prior.Prior for each non-fixed param

    lik_variance
        Variance of the observation (Gaussian) likelihood

    lik_variance_fixed
        whether the likelihood variance is fixed or can be optimized

    opt_params
        ['method'] = 'grad', 'multigrad', 'direct', 'slice'

        ['bounds'] = bounds for hps

        DIRECT:
           ['n_direct_evals'] = (DIRECT) number of iterations

        MULTIGRAD:
            ['num_restarts'] = (multigrad) num of restarts

            ['options'] = options to be passed to minimize()
            e.g. {'maxiters': 100}

        SLICE:
            ['n_samples'] = number of slice samples to generate

    remove_y_mean
        Boolean whether to remove the mean of y in the model. This is added
        on when predicting.

    auto_update
        If true, then self.update() is run after data or parameters change.
        Otherwise self.update() needs to be called manually.

    stabilise_mat_inv
        whether a small amount is added to diagonal of K to help invert it.
        This is only done if the normal inverse gave a LinAlgError

    verbose
        verbose level
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray,
                 kern: Optional[GPy.kern.Kern] = None,
                 lik_variance: Optional[float] = None,
                 lik_variance_fixed: Optional[bool] = False,
                 hyper_priors: Optional[List] = None,
                 kernel_params_fixed: Optional[bool] = False,
                 opt_params: Optional[dict] = None,
                 remove_y_mean: Optional[bool] = False,
                 y_norm: Optional[str] = 'mean',
                 stabilise_mat_inv: Optional[bool] = True,
                 auto_update: Optional[bool] = True,
                 verbose: Optional[Union[bool, int]] = False) -> None:
        assert X.ndim == 2 and Y.ndim == 2
        assert len(X) == len(Y)

        # Whether to automatically update after things change
        # If this __init__() is called from a child class, then the child's
        # self.auto_update will overwrite this to the correct value
        self.auto_update = auto_update

        self.verbose = verbose
        if remove_y_mean:
            self.y_norm = 'mean'
            print("Stop using remove_y_mean!")
        else:
            self.y_norm = y_norm  # 'mean', 'meanstd'

        self.stabilise_mat_inv = bool(stabilise_mat_inv)

        self.lik_variance_fixed = lik_variance_fixed

        if kern is None:
            self.kern = GPy.kern.RBF(X.shape[1])
        else:
            self.kern = kern
        if verbose:
            print("Set kernel to\n", self.kern)

        self.n_kern_params = len(self.kern.param_array)

        self.default_opt_params = {'method': 'grad'}

        if opt_params is None:
            self.opt_params = {'method': 'grad'}
        else:
            self.opt_params = opt_params

        if verbose:
            print("opt_params = ", self.opt_params)

        if lik_variance is not None:
            self._lik_variance = lik_variance
        else:
            self._lik_variance = 0.1
        self._lik_variance_gradient = None

        self.kernel_params_fixed = kernel_params_fixed

        if hyper_priors is not None:
            assert len(hyper_priors) == len(
                self.param_array), "Need to provide a prior for each hp!"
        self.hyper_priors = hyper_priors

        # intialise the log likelihood and derivatives
        self.ll = None
        self.dL_dK = None
        self.Ka = None
        self.Ka_inv = None
        self.alpha = None
        self.Chol_Ka = None
        self.Chol_Ka_inv = None
        self.LogDetKa = None

        self.X = None
        self.Y = None
        self.Y_raw = None
        self.y_mean = None
        self.y_std = None
        self.output_dim = None

        self.set_data(X, Y, update=False)

        # Using the parameter passed in because this avoids running the
        # basic_gp update() if I'm running __init__() from a child class
        if auto_update:
            self.update()

    def __repr__(self) -> str:
        if self.opt_params['method'] not in ['slice']:
            s = "GP with kernel \n"
            s += self.kern.__str__() + "\n\n"
            s += "lik_variance = " + str(self._lik_variance) + "\n"
            s += "param_array = " + str(self.param_array) + "\n"
            s += "X.shape = " + str(self.X.shape) + "\n"
            s += "Y.shape = " + str(self.Y.shape) + "\n"
            s += "Objective (marginal joint) = " + str(-self.objective()) + "\n"
        else:
            s = "GP with " + self.kern.name + " kernel\n"
            s += "Inference method: " + self.opt_params['method'] + "\n"
            s += "X.shape = " + str(self.X.shape) + "\n"
            s += "Y.shape = " + str(self.Y.shape)

        return s

    @property
    def param_array(self) -> np.ndarray:
        """The parameter array. [kernel hps, lik_variance]

        Takes into account which hps are fixed and doesn't return those. This
        makes the optimisation using gradient descent/DIRECT easier to
        deal with.

        Returns
        -------
        np.ndarray with the free params
        """
        # If a param is fixed, then the empty array will be ignored in the
        # hstack operation at the end
        if self.kernel_params_fixed:
            kp = np.array([])
        else:
            kp = self.kern.param_array

        if self.lik_variance_fixed:
            lik = np.array([])
        else:
            lik = self._lik_variance

        return np.hstack((kp, lik))

    @param_array.setter
    def param_array(self, p: np.ndarray) -> None:
        """Update the param_array.

        Only updates hps that are not fixed. If fixed hps need to be
        changed, they can be accessed directly (e.g. self.lik_variance)

        Parameters
        ----------
        p
            New param array
        """

        # Check that p is the right size
        n = 0
        if not self.lik_variance_fixed:
            n += 1
        if not self.kernel_params_fixed:
            n += self.n_kern_params
        assert len(p) == n, "Wrong size of param_array"

        # Pop each relevant slice depending on which params are fixed
        _p = p.copy()

        if not self.lik_variance_fixed:
            _p, self._lik_variance = _p[:-1], _p[-1]

        if not self.kernel_params_fixed:
            self.kern[:] = _p

        if self.auto_update:
            self.update()

    @property
    def gradient(self) -> np.ndarray:
        """Returns an array of the gradients of the likelihood wrt the
        kernel hps and the likelihood variance Takes into account which hps
        are fixed and doesn't return those gradients.

        Returns
        -------
        np.ndarray with the gradients of the free params

        """
        # If a param is fixed, then the empty array will be ignored in the
        # hstack operation at the end
        if self.kernel_params_fixed:
            kp = np.array([])
        else:
            kp = self.kern.gradient

        if self.lik_variance_fixed:
            lik = np.array([])
        else:
            lik = self._lik_variance_gradient

        return np.hstack((kp, lik))

    def set_data(self, X: np.ndarray = None, Y: np.ndarray = None,
                 update: bool = True) -> None:
        """Sets the fields that are provided

        NOTE: update parameter has no effect. Leaving it in for now in case
        it is used somewhere.

        Parameters
        ----------
        X
            new X data

        Y
            new Y data

        update
            (DOES NOTHING)
        """
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y_raw

        assert len(X) == len(Y)
        self.X, self.Y_raw = X.copy().astype(float), Y.copy().astype(float)

        self.y_mean = np.mean(Y)
        self.y_std = np.std(Y)
        if self.y_norm == 'mean':
            self.Y = self.Y_raw - self.y_mean
            if self.verbose > 1:
                print("set_data(): removing mean of Y from data")
        elif self.y_norm == 'meanstd':
            if self.y_std == 0:  # fix for when all y are the same
                self.Y = (self.Y_raw - self.y_mean) / 1.0
            else:
                self.Y = (self.Y_raw - self.y_mean) / self.y_std

        else:
            self.Y = self.Y_raw

        self.output_dim = Y.shape[1]

        if self.auto_update:
            self.update()

    def set_XY(self, X: np.ndarray = None, Y: np.ndarray = None,
               update: bool = True) -> None:
        """GPy interface uses this name. This runs self.set_data()"""

        self.set_data(X=X, Y=Y, update=update)

    def update(self) -> None:
        """Update things

        Called when params and/or data are changed
        """
        # # requires "import inspect" at the top of the file"
        # curframe = inspect.currentframe()
        # calframe = inspect.getouterframes(curframe, 2)
        # print "Update() called from", calframe[1][3]

        self.Ka = self.compute_Ka()

        # If we want to try to stabilise the inverse, then try adding
        # 1e-8 along the diagonal if we have LinAlgError
        if self.stabilise_mat_inv:
            try:
                self.Ka_inv, self.Chol_Ka, \
                self.Chol_Ka_inv, self.LogDetKa = \
                    GPy.util.linalg.pdinv(self.Ka)
            except (np.linalg.LinAlgError, sp.linalg.LinAlgError):
                self.Ka += 1e-8 * np.eye(len(self.Ka))
                try:
                    self.Ka_inv, self.Chol_Ka, \
                    self.Chol_Ka_inv, self.LogDetKa = \
                        GPy.util.linalg.pdinv(self.Ka)
                except (np.linalg.LinAlgError, sp.linalg.LinAlgError):
                    if self.verbose:
                        print("K is singular! Crashed model's summary:")
                        print(self)
                    raise np.linalg.LinAlgError("Matrix singular despite fix!")
        else:
            self.Ka_inv, self.Chol_Ka, self.Chol_Ka_inv, self.LogDetKa = \
                GPy.util.linalg.pdinv(self.Ka)

        # alpha is used for predictions
        self.alpha, _ = GPy.util.linalg.dpotrs(self.Chol_Ka, self.Y, lower=1)

        # update gradients by computing dL_dK and running the kernel's method
        self.dL_dK = self.compute_dL_dK()
        self.kern.update_gradients_full(self.dL_dK, self.X, self.X)

        # I think this is correct
        self._lik_variance_gradient = np.trace(self.dL_dK)

        # Update the likelihood
        if self.opt_params['method'] not in ['slice']:
            self.ll = self.log_likelihood()

    def log_prior(self):
        log_prior = 0
        if self.hyper_priors is not None:
            params = self.param_array
            for ii, param in enumerate(params):
                log_prior += self.hyper_priors[ii].lnpdf(param)
        return log_prior

    def log_prior_gradient(self):
        if self.hyper_priors is not None:
            params = self.param_array
            log_prior_grad = np.zeros(len(params))
            for ii, param in enumerate(params):
                log_prior_grad[ii] = self.hyper_priors[ii].lnpdf_grad(param)
        else:
            log_prior_grad = 0

        return log_prior_grad

    def objective(self, theta: np.ndarray = None) -> float:
        """Objective function provided to optimiser.

        Parameters
        ----------
        theta
            Parameter array at which to evaluate the log likelihood

        Returns
        -------
        float log likelihood
        """
        if theta is not None:
            self.param_array = theta.flatten()
        result = -self.log_likelihood() - self.log_prior()
        if self.verbose:
            print("Objective function")
            print("param_array", self.param_array)
            print("objective", result)
        return result

    def objective_grad(self, theta: np.ndarray = None) -> np.ndarray:
        """Gradients of the likelihood w.r.t. theta

        Parameters
        ----------
        theta
            Parameter array at which to evaluate the gradient

        Returns
        -------
        gradient as np.ndarray
        """
        if theta is not None:
            self.param_array = theta.flatten()

        # self.gradient is already the grad of the NLL,
        # so only log_prior_gradient needs to be negated
        grad = self.gradient + self.log_prior_gradient()
        if self.verbose:
            print("Objective gradient function")
            print("param_array", self.param_array)
            print("gradient", grad)
        return grad

    def objective_log_theta(self, log_theta: np.ndarray) -> float:
        """Objective function provided to optimiser.

        Parameters
        ----------
        log_theta
            Log(theta) at which to evaluate the objective

        Returns
        -------
        log likelihood at log(theta)
        """
        return self.objective(np.exp(log_theta).flatten())

    def objective_grad_log_theta(self, log_theta: np.ndarray) -> np.ndarray:
        """Gradients of the likelihood w.r.t. log(theta)

        Parameters
        ----------
        log_theta
            Log(theta) at which to evaluate the gradient

        Returns
        -------
        gradient as np.ndarray
        """
        return -self.param_array * self.objective_grad(
            np.exp(log_theta).flatten())

    def optimize(self, opt_params: dict = None,
                 verbose: Union[int, bool] = False) -> OptimizeResult:
        """Optimize function

        For now only allowing positive hps for gradient descent by running
        the optimisation in log space.

        DIRECT and slice are bounded by the opt_params['hp_bounds'] field.

        opt_params allows us to provide a different set of optimization

        Parameters
        ----------
        opt_params
            Dictionary with the optimisation params. More details in the
            class docstring.

        verbose

        Returns
        -------
        OptimizeResult object
        """
        # Avoids UnboundLocalError by defining 'res' in case the
        # optimisation procedure doesn't have an obvious return object
        # e.g. slice sampling
        res = None
        if opt_params is None:
            opt_params = self.opt_params
        elif opt_params == 'default':
            opt_params = self.default_opt_params

        # Hacky check of verbosity: checking opt_params and function input.
        # Casting to bool in case one of the values is None
        if 'verbose' in opt_params.keys():
            verbose = bool(verbose or opt_params['verbose'])

        if opt_params['method'] == 'grad':
            # Normal gradient descent is done in log(hp) space
            if 'options' in opt_params.keys():
                options = opt_params['options']
            else:
                options = None

            res = sp.optimize.minimize(self.objective_log_theta,
                                       np.log(self.param_array),
                                       jac=self.objective_grad_log_theta,
                                       options=options)

            new_param_array = np.exp(res.x)
            self.param_array = new_param_array
            if verbose:
                print("Finished grad descent optimization of hps")
                print("Result:")
                pprint(res)
                print("New model")
                print(self)
            return res

        elif opt_params['method'] == 'direct':
            if verbose:
                print("Starting DIRECT optimization of hps")
                print("DIRECT options:")
                pprint(opt_params)

            assert 'hp_bounds' in opt_params.keys()
            assert 'n_direct_evals' in opt_params.keys()

            # hp_bounds are in hp-space, so transform them to log space
            hp_bounds = np.log(opt_params['hp_bounds'])

            res = scipydirect.minimize(self.objective,
                                       hp_bounds,
                                       maxf=opt_params['n_direct_evals'])
            self.param_array = np.exp(res.x)
            if verbose:
                print("Finished DIRECT optimization of hps")
                print("Result:")
                pprint(res)
                print("New model")
                print(self)

            return res

        elif opt_params['method'] == 'multigrad':
            if 'options' in opt_params.keys():
                options = opt_params['options']
            else:
                options = None

            assert 'restart_bounds' in opt_params.keys()
            # bounds are in hp-space, so transform them to log space
            restart_bounds = np.log(opt_params['restart_bounds'])

            if 'hp_bounds' in opt_params.keys():
                hp_bounds = np.log(opt_params['hp_bounds'])
            else:
                hp_bounds = None

            current_param_array = self.param_array.copy()
            num_restarts = opt_params['num_restarts']
            res = minimize_with_restarts(self.objective_log_theta,
                                         restart_bounds,
                                         num_restarts=num_restarts,
                                         jac=self.objective_grad_log_theta,
                                         hard_bounds=hp_bounds,
                                         minimize_options=options,
                                         verbose=verbose)

            # if multi-started gradient descent failed, then rollback
            if res is None:
                self.param_array = current_param_array
            else:
                new_param_array = np.exp(res.x)
                self.param_array = new_param_array
            if verbose:
                print("Finished multigrad descent optimization of hps")
                print("Result:")
                pprint(res)
                print("New model")
                print(self)
            return res

        elif opt_params['method'] == 'samplegrad':
            if 'minimize_options' in opt_params.keys():
                minimize_options = opt_params['minimize_options']
            else:
                minimize_options = None

            if 'num_samples' in opt_params.keys():
                num_samples = opt_params['num_samples']
            else:
                num_samples = 1000
            if 'num_local' in opt_params.keys():
                num_local = opt_params['num_local']
            else:
                num_local = 5

            hp_bounds = np.log(opt_params['hp_bounds'])

            res = sample_then_minimize(
                self.objective_log_theta,
                hp_bounds,
                num_samples=num_samples,
                num_local=num_local,
                jac=self.objective_grad_log_theta,
                minimize_options=minimize_options,
                evaluate_sequentially=True,
                verbose=False)

            self.param_array = np.exp(res.x)


        elif opt_params['method'] == 'slice':
            print("Running optimize with slice sampling inside a model class!",
                  "Do this in a model_collection instead!")
            raise NotImplementedError

        else:
            print("Bad optimiser choice")
            raise NotImplementedError

        # return res

    def predict(self, x_star: np.ndarray, y_star: np.ndarray = None,
                full_cov: bool = False, **kwargs) -> Tuple:
        """Prediction of mean and variance.

        If y_star is given, then the log probabilities of the posterior are
        also provided as the third output

        If slice sampling, then each instance of the slice hps is applied and
        their average predictions (mean and variance) are returned. This can
        be quite slow, so use the sampling model-wrapper in model_collection.py
        for a slightly faster, but more memory-intensive implementation.

        Parameters
        ----------
        x_star
            Locations to predict at

        y_star
            (optional) True values of y at x_star

        full_cov
            whether we want the full covariance

        Returns
        -------
        (predicted mean, predicted var (, log likelihood))
        """
        mu, var = self.predict_latent(x_star, full_cov=full_cov, **kwargs)
        # add observation noise
        var += self._lik_variance

        if y_star is not None:
            log_prob = -0.5 * (np.log(2 * np.pi) + np.log(var) +
                               (y_star - mu) ** 2 / var)
            return mu, var, log_prob
        return mu, var

    def predict_latent(self, x_star: np.ndarray,
                       full_cov: bool = False, kern=None) -> \
            Tuple[np.ndarray, np.ndarray]:
        """latent function prediction

        Parameters
        ----------
        x_star
            Locations to predict at

        full_cov
            whether we want the full covariance

        kern
            Optional if we want to predict with a different kernel
            (e.g. for looking at subspaces of a combination kernel)

        Returns
        -------
        (predicted mean, predicted var (, log likelihood))

        """
        assert x_star.ndim == 2

        if kern is None:
            kern = self.kern

        k_star = kern.K(x_star, self.X)
        k_star_star = kern.K(x_star, x_star)

        mu = k_star.dot(self.alpha)

        var = k_star_star - k_star.dot(self.Ka_inv.dot(k_star.T))
        # Need to copy, else np.diag returns read-only array
        if not full_cov:
            var = np.diag(var).copy().reshape(mu.shape)

        # If the y data has been transformed to zero-mean,
        # then undo that here
        if self.y_norm == 'mean':
            mu += self.y_mean
        elif self.y_norm == 'meanstd':
            mu = mu * self.y_std + self.y_mean
            var = var * self.y_std ** 2
        return mu, var

    def compute_Ka(self, X: np.ndarray = None,
                   X2: np.ndarray = None) -> np.ndarray:
        """Returns the complete covariance matrix (kernel + lik).

        If X is not provided, then the GP's own data is used.

        Parameters
        ----------
        X
            (optional) input locations

        X
            (optional) input locations

        Returns
        -------
        k(X, X2)
        """
        if X is None:
            X = self.X

        Ka = self.kern.K(X, X2=X2) + self._lik_variance * np.eye(len(X))

        return Ka

    def log_likelihood(self, Ka_inv: np.ndarray = None,
                       LogDetKa: float = None) -> float:
        """log likelihood computation.

        Parameters
        ----------
        Ka_inv
            (optional) Inverse of Ka

        LogDetKa
            (optional) log(det(Ka))

        Returns
        -------
        marginal log likelihood of the GP
        """
        # if self.mode == 'slice':
        #   print("log_likelihood() only meant for use in self.mode = normal!")
        if Ka_inv is None:
            Ka_inv = self.Ka_inv

        if LogDetKa is None:
            LogDetKa = self.LogDetKa

        ll = -0.5 * (len(self.X) * np.log(2 * np.pi) +
                     LogDetKa +
                     self.Y.T.dot(Ka_inv.dot(self.Y)))

        # ll = ll[0, 0]

        return ll.item()

    def compute_dL_dK(self, Ka_inv: np.ndarray = None,
                      Chol_Ka: np.ndarray = None, alpha: np.ndarray = None) \
            -> np.ndarray:
        """Compute the derivative of the log likelihood w.r.t. the covariance
        matrix K.

        This is passed to the update_gradients function of the kernel to
        compute the gradients wrt theta.

        If an arg is not passed to the function, the self.arg is used.

        Parameters
        ----------
        Ka_inv
            (optional) Inverse of Ka

        Chol_Ka
            (optional) Cholesky decomposition of Ka

        alpha
            (optional) alpha term (see e.g. Rasmussen's book Ch. 2)

        Returns
        -------
        dL/dK
        """
        if Chol_Ka is None:
            Chol_Ka = self.Chol_Ka
        if Ka_inv is None:
            Ka_inv = self.Ka_inv
        if alpha is None:
            alpha = self.alpha

        # alpha, _ = GPy.util.linalg.dpotrs(Chol_Ka, self.Y, lower=1)

        return 0.5 * (alpha.dot(alpha.T) - Ka_inv)

    def dmu_dx(self, x_star: np.ndarray) -> np.ndarray:
        """
        Returns the derivative of the posterior mean evaluated at locations
        x_star.

        Keeping this function as this interface is used in some old code.

        Parameters
        ----------
        x_star

        Returns
        -------
        dmu/dx at x_star
        """
        return self.dposterior_dx(x_star)[0]

    def dposterior_dx(self, x_star: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Computes the gradient of the posterior

        Not the same as the mean and variance of the derivative!

        Somewhat clunky code because it's adapted from GPy directly.

        Parameters
        ----------
        x_star
            Points to evaluate the derivative at

        Returns
        -------
        (dmu_dx, dvar_dx) at x_star
        """
        kern = self.kern
        woodbury_vector = self.alpha
        woodbury_inv = self.Ka_inv
        mean_jac = np.empty(
            (x_star.shape[0], x_star.shape[1], self.output_dim))
        for i in range(self.output_dim):
            mean_jac[:, :, i] = \
                kern.gradients_X(woodbury_vector[:, i:i + 1].T,
                                 x_star,
                                 self.X)

        # Gradients wrt the diagonal part k_{xx}
        dv_dX = kern.gradients_X_diag(np.ones(x_star.shape[0]), x_star)

        var_jac = dv_dX
        alpha = -2. * np.dot(kern.K(x_star, self.X),
                             woodbury_inv)

        var_jac += kern.gradients_X(alpha, x_star, self.X)

        return mean_jac, var_jac

    # @classmethod
    def plot(self, model=None, n: int = None, x=None, eps: float = None,
             cmap: matplotlib.colors.Colormap = 'YlOrRd',
             title: str = None, ylim: np.ndarray = None,
             return_fig_handle: bool = False) \
            -> Union[None, plt.Figure]:
        """Plot the model (data, mean and variance)

        Optionally provide a model class. This uses all relevant info
        from this class to create the plot. Using this for model_collection
        objects.

        Parameters
        ----------
        model
            (optional) provides alternative model class to plot

        n
            number of points to evaluate for the plot. Default 1D = 200,
            2D = 75 in each direction

        x
            vector of x locations to plot (only works on 1D for now)

        eps
            how much to extend over the limits of the training data. Default
            0.05.

        cmap
            matplotlib colormap

        title
            Title string of the plot

        ylim
            [min, max] of y-axis

        return_fig_handle
            whether to return the figure handle
        """
        # Hack to allow wrapper models to use this function until I fix this
        if model is not None:
            self = model

        # how much to extend past the limits of the training data
        if eps is None:
            eps = 0.05

        x_range = np.max(self.X) - np.min(self.X)

        if self.X.shape[1] == 1:  # 1D
            # resolution
            if n is None:
                n = 200
            if x is None:
                x_plot = np.linspace(min(self.X) - eps * x_range,
                                     max(self.X) + eps * x_range,
                                     n).flatten()
            else:
                x_plot = x.flatten()
            mu, var = self.predict(x_plot[:, None])
            mu, var = mu.flatten(), var.flatten()
            if return_fig_handle:
                f = plt.figure()
            plt.plot(x_plot, mu, 'g')
            plt.fill_between(x_plot, mu,
                             mu + 2 * np.sqrt(var), alpha=0.4)
            plt.fill_between(x_plot, mu,
                             mu - 2 * np.sqrt(var), alpha=0.4)
            plt.plot(self.X[:, 0], self.Y[:, 0] + self.y_mean, 'b*')
            if title is None:
                plt.title('{} with {} kernel'.format(self.__class__.__name__,
                                                     self.kern.name))
            else:
                plt.title(title)

            if ylim is not None:
                plt.ylim(ylim)

            if return_fig_handle:
                return f

        elif self.X.shape[1] == 2:  # 2D
            # resolution
            if n is None:
                n = 75

            # get the limits
            x1_min, x1_max = np.min(self.X[:, 0]), np.max(self.X[:, 0])
            x1_range = x1_max - x1_min
            x2_min, x2_max = np.min(self.X[:, 1]), np.max(self.X[:, 1])
            x2_range = x2_max - x2_min

            x1_min, x1_max = x1_min - x1_range * eps, x1_max + x1_range * eps
            x2_min, x2_max = x2_min - x2_range * eps, x2_max + x2_range * eps

            # Create the arrays
            x1 = np.linspace(x1_min, x1_max, n)
            x2 = np.linspace(x2_min, x2_max, n)
            x1_mesh, x2_mesh = np.meshgrid(x1, x2)

            # Get predictions over the range of X
            y, y_var = self.predict(np.hstack((x1_mesh.reshape(-1, 1),
                                               x2_mesh.reshape(-1, 1))))
            y = y[:, 0].reshape(n, n)
            y_var = y_var[:, 0].reshape(n, n)

            # plot the mean and variance in two subplots
            f, (ax1, ax2) = plt.subplots(2, 1,
                                         figsize=(8, 12))

            # mean
            y_plot = ax1.contourf(y, extent=[x1_min,
                                             x1_max,
                                             x2_min,
                                             x2_max],
                                  cmap=cmap)
            plt.colorbar(y_plot, ax=ax1)
            ax1.plot(self.X[:, 0], self.X[:, 1], '.', markersize=20)
            ax1.set_title('mean')
            ax1.set_xlim([x1_min,
                          x1_max])
            ax1.set_ylim([x2_min,
                          x2_max])

            # variance
            y_var_plot = ax2.contourf(y_var,
                                      extent=[x1_min,
                                              x1_max,
                                              x2_min,
                                              x2_max],
                                      cmap=cmap)
            plt.colorbar(y_var_plot, ax=ax2)
            ax2.plot(self.X[:, 0], self.X[:, 1], '.', markersize=20)
            ax2.set_title('variance')
            ax2.set_xlim([x1_min,
                          x1_max])
            ax2.set_ylim([x2_min,
                          x2_max])

            if title is None:
                f.suptitle('{} with {} kernel'.format(self.__class__.__name__,
                                                      self.kern.name))
            else:
                f.suptitle(title)

            # Tight layout reduces the space between suptitle and the rest
            f.tight_layout(rect=[0, 0.03, 1, 0.97])

            if return_fig_handle:
                return (f, (ax1, ax2))

        else:
            raise NotImplementedError
