import numpy as np
import george
import emcee
import scipy.stats


class GPFit:
    def __init__(self, time_stamps, flux, flux_error, kernel_long):
        self.time = time_stamps
        self.flux = flux
        self.flux_err = flux_error
        self.kernel_long = kernel_long
        self.data_pts = len(time_stamps)
        self.params = None
        self.walker_params = None
        self.gp = None
        self.sampler = None
        self.lsp_period = None

    def set_params(self):
        """Calculates initial gp parameter values based on data."""

        mean_flux = np.mean(self.flux)

        # kernel1: long-term phase change
        log_amp_k1 = np.log(self.flux.max() - self.flux.min())
        log_metric = np.log(5 ** 2)

        # kernel2: short-term periodic
        log_amp_k2 = np.log(self.flux.max() - self.flux.min())
        gamma = 10  # ~10 nornal gamma
        log_period = 0

        if self.kernel_long:
            parameters = {
                "mean": mean_flux,
                "log_amp_k1": log_amp_k1,
                "log_metric": log_metric,
                "log_amp_k2": log_amp_k2,
                "gamma": gamma,
                "log_period": log_period,
            }
        else:
            parameters = {
                "mean": mean_flux,
                "log_amp_k2": log_amp_k2,
                "gamma": gamma,
                "log_period": log_period,
            }

        self.params = parameters
        return

    def set_walker_param_matrix(self, nwalkers):
        """Creates a matrix of starting parameters for every walker."""

        if self.params is not None:
            # create an array of all parameter values
            p_start = np.array(list(self.params.values()))
            cov_matrix = np.sqrt(np.diag(p_start) ** 2)
            p0 = np.random.multivariate_normal(
                mean=p_start, cov=cov_matrix, size=(nwalkers)
            )

            # randomly distributed starting period values from ~1hr to ~24hrs
            p0[:, -1] = np.random.normal(size=nwalkers) * 0.5 + np.log(4 / 24.0)

            print(p0[:, -2])

            self.walker_params = p0

        else:
            print("Please set parameter values first.")

        return

    def set_gp_kernel(self):
        """Sets up the Gaussian Process Kernel that is needed for george."""

        if self.kernel_long:
            k1 = np.exp(self.params["log_amp_k1"]) * george.kernels.ExpSquaredKernel(
                metric=np.exp(self.params["log_metric"])
            )

        k2 = np.exp(self.params["log_amp_k2"]) * george.kernels.ExpSine2Kernel(
            gamma=(self.params["gamma"]), log_period=self.params["log_period"]
        )

        if self.kernel_long:
            kernel = k1 * k2
        else:
            kernel = k2

        gp = george.GP(kernel, fit_mean=True, mean=self.params["mean"])
        gp.compute(self.time, self.flux_err)

        self.gp = gp

        return

    def run_emcee(self, nwalkers, niter, threads, burn_in):
        """Runs emcee's mcmc code."""

        ndim = self.gp.vector_size  #ndim is 4 if not including longterm kernel
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            post_lnlikelihood,
            args=[self.gp, self.time, self.flux, self.flux_err],
            threads=threads,
        )

        # run steps for a burn-in
        state = sampler.run_mcmc(self.walker_params, burn_in)
        sampler.reset()
        sampler.run_mcmc(state[0], niter)
        self.sampler = sampler

        return sampler


def prior(params):

    """
    Calculated the log of the prior values, given parameter values.

    Parameters
    ----------
    params : list
        List of all kernel parameters

    param[0] : float
        mean (between 0 and 2)

    param[1] : float
        k1 log constant

    param[2] : float
        k1 log metric

    param[3] : float
        k2 log amplitude (between -10 and 10)

    param[4] : float
        log gamma (log gamma between ####0.1 and 40)

    param[5] : float
        log period (period between ####1h and 24hrs)

    Returns
    -------
    sum_log_prior : int
        sum of all log priors (-inf if a parameter is out of range)

    """
    # TODO: Improve documentation for prior ranges

    if len(params) == 6:

        p_mean = scipy.stats.norm(1, 0.5).logpdf(params[0])

        p_log_amp_k1 = scipy.stats.norm(np.log(2), np.log(10)).logpdf(params[1])
        p_log_metric = scipy.stats.norm(np.log(100), np.log(10)).logpdf((params[2]))

        p_log_amp_k2 = scipy.stats.norm(np.log(2), np.log(2)).logpdf(params[3])
        p_log_gamma = scipy.stats.norm(np.log(10), np.log(2)).logpdf(np.log(params[4]))
        p_log_period = scipy.stats.norm(np.log(4.0 / 24.0), (12.0 / 24.0)).logpdf(params[5])

        sum_log_prior = (
            p_mean
            + p_log_amp_k1
            + p_log_metric
            + p_log_amp_k2
            + p_log_gamma
            + p_log_period
        )

    else:

        p_mean = scipy.stats.norm(1, 0.5).logpdf(params[0])

        p_log_amp_k2 = scipy.stats.norm(np.log(2), np.log(2)).logpdf(params[1])
        p_log_gamma = scipy.stats.norm(np.log(10), np.log(2)).logpdf(np.log(params[2]))
        p_log_period = scipy.stats.norm(np.log(4.0 / 24.0), (12.0 / 24.0)).logpdf(params[3])

        sum_log_prior = p_mean + p_log_amp_k2 + p_log_gamma + p_log_period

    if np.isnan(sum_log_prior) is True:
        return -np.inf

    return sum_log_prior


def logl(params, gp, tsample, fsample, flux_err):
    """
    Compute log likelihood based on given parameters.
    """

    gp.set_parameter_vector(params)

    try:
        gp.compute(tsample, flux_err)
        lnlike = gp.lnlikelihood(fsample)
        
    except np.linalg.LinAlgError:
        lnlike = -1e25

    return lnlike


def post_lnlikelihood(params, gp, tsample, fsample, flux_err):

    """
    Calculates the posterior likelihood from the log prior and
    log likelihood.

    Parameters
    ----------
    params : list
        List of all kernel parameters

    Returns
    -------
    ln_likelihood : float
        The posterior, unless the posterior is infinite, in which case,
        -1e25 will be returned instead.

    """

    # calculate the log_prior
    log_prior = prior(params)

    # return -inf if parameters are outside the priors
    if np.isneginf(log_prior) is True:
        return -np.inf

    try:
        lnlike = logl(params, gp, tsample, fsample, flux_err)
        ln_likelihood = lnlike + log_prior

    except np.linalg.linalg.LinAlgError:
        ln_likelihood = -1e25

    return ln_likelihood if np.isfinite(ln_likelihood) else -1e25
