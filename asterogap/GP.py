import numpy as np
import george
import emcee
import scipy.stats
import pandas as pd
import os


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

        self.kde = log_period_prior_kde()


    def set_params(self):
        """Calculates initial gp parameter values based on data."""

        mean_flux = np.mean(self.flux)

        # kernel1: long-term phase change
        log_amp_k1 = np.log(self.flux.max() - self.flux.min())
        log_metric = np.log(5 ** 2)

        # kernel2: short-term periodic
        log_amp_k2 = np.log(self.flux.max() - self.flux.min())
        gamma = 10  # ~10 normal gamma
        log_period = 0 # 24 hours

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

    def set_walker_param_matrix(self, nwalkers, seed=None):
        """Creates a matrix of starting parameters for every walker."""

        if self.params is not None:
            # create an array of all parameter values
            p_start = np.array(list(self.params.values()))
            #cov_matrix = np.sqrt(np.diag(p_start) ** 2)

            p0 = scipy.stats.multivariate_normal.rvs(
                p_start, size=nwalkers, random_state=seed
            )

            # p0 = np.random.multivariate_normal(
            #     mean=p_start, cov=cov_matrix, size=(nwalkers)
            # )

            # randomly distributed starting period values from ~1hr to ~24hrs
            #np.random.seed(seed)

            #p0[:, -1] = np.random.normal(size=nwalkers) * 0.5 + np.log(4 / 24.0)

			#calculate kde for period
            #kde = log_period_prior_kde()
            #x_grid = np.linspace(-8, 5, 1000) #min should be 30 minutes
            #pdf = kde.evaluate(x_grid)
            p0_samples = self.kde.resample(nwalkers)

            print(p0_samples)

            p0[:, -1] =  p0_samples

            # print(p0[:, -2])

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
            args=[self.gp, self.time, self.flux, self.flux_err, self.kde],
            threads=threads,
        )

        # run steps for a burn-in
        state = sampler.run_mcmc(self.walker_params, burn_in)

        print(state.coords)
        sampler.reset()
        sampler.run_mcmc(state.coords, niter)
        self.sampler = sampler

        return sampler


def prior(params, kde):

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

        #p_log_amp_k1 = scipy.stats.norm(np.log(2), np.log(10)).logpdf(params[1])
        p_log_amp_k1 = scipy.stats.uniform(-10, 20).logpdf(params[1])
        p_log_metric = scipy.stats.norm(np.log(100), np.log(10)).logpdf((params[2]))

        p_log_amp_k2 = scipy.stats.norm(np.log(2), np.log(2)).logpdf(params[3])
        p_log_gamma = scipy.stats.norm(np.log(10), np.log(2)).logpdf(np.log(params[4]))

        #kde = log_period_prior_kde()
        #if period is smaller than 1 minute or larger than a year, then -np inf
        if params[5] < -7.27:
            p_log_period = -np.inf
#        if params[5] < -700:
#            print("v smol %d" %params[5])
#            p_log_period = -np.inf
        elif params[5] > 5.9:
            p_log_period = -np.inf
        else:
        	p_log_period = kde.evaluate(params[5]) #scipy.stats.norm(np.log(4.0 / 24.0), (12.0 / 24.0)).logpdf(params[5])


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
        #kde = log_period_prior_kde()
        #if period is smaller than 1 minute, then -np inf
        if params[5] < -7.27:
            p_log_period = -np.inf
        if params[5] > 5.9:
            p_log_period = -np.inf
        else:
        	p_log_period = kde.evaluate(params[5]) #scipy.stats.norm(np.log(4.0 / 24.0), (12.0 / 24.0)).logpdf(params[5])

        sum_log_prior = p_mean + p_log_amp_k2 + p_log_gamma + p_log_period

    if np.isnan(sum_log_prior) == True:
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


def post_lnlikelihood(params, gp, tsample, fsample, flux_err, kde):

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
    log_prior = prior(params, kde)

    # return -inf if parameters are outside the priors
    if np.isneginf(log_prior) == True:
        return -np.inf

    try:
        lnlike = logl(params, gp, tsample, fsample, flux_err)
        ln_likelihood = lnlike + log_prior

    except np.linalg.linalg.LinAlgError:
        ln_likelihood = -1e25

    return ln_likelihood if np.isfinite(ln_likelihood) else -1e25

def read_lcdb(filename='LC_SUM_PUB.TXT'):
    """Read LCDB file tabulating periods and associated information.
    LCDB available from:
    http://www.minorplanet.info/lightcurvedatabase.html
    Please see that page for citation information.
    Warner, B.D., Harris, A.W., Pravec, P. (2009). Icarus 202, 134-146.
    Updated <Date of last update>. http://www.MinorPlanet.info/lightcurvedatabase.html
    The Asteroid Lightcurve Database is a listing of asteroid lightcurve parameters and other information,
    e.g., estimated/measured diameters, absolute magnitudes (H), phase slope parameters (G), albedos, + more.
    Parameters
    ----------
    filename : str, opt
        The full path to the LCDB file (LC_SUM_PUB.TXT or similarly formatted files)
    Returns
    -------
    pandas.DataFrame
    """
    names = ['numberId', 'new', 'Name', 'Desig', 'family', 'Csource', 'class', 'Dsource',
            'Dflag', 'diameter','Hsource', 'H', 'Hband', 'Asource', 'Albedoflag',
            'albedo', 'Pflag', 'period', 'Pdescr','Ampflag', 'AmpMin', 'AmpMax',
            'U', 'notes', 'binary', 'priv' , 'pole', 'survA']
    colspecs = [(0, 7), (8, 9), (10, 40), (41, 61), (62, 70), (71, 72), (73, 78),
                (79, 80), (81, 82), (83, 92), (93, 94), (95, 100), (101, 102),
                (104, 105), (106, 107), (108, 114), (115, 116), (117, 130), (131, 146),
                (147, 148), (149, 153), (154, 158), (159, 161),(162, 167), (168, 171),
                (172, 175), (176, 179), (180, 182), (183, 184)]
    lcdata = pd.read_fwf(filename, index_col=False, skiprows=5, names=names, colspecs=colspecs)
    objId = lcdata.query('numberId == numberId')['numberId'].astype('int').astype('str')
    objId2 = lcdata.query('numberId != numberId and Desig == Desig')['Desig']
    objId3 = lcdata.query('numberId != numberId and Desig != Desig')['Name']
    t = pd.concat([objId, objId2, objId3])
    lcdata = lcdata.merge(t.to_frame(name='objId'), left_index=True, right_index=True)
    lcdata['Frequency'] = 1.0/lcdata.period
    # Map flags to ints.
    tmp = lcdata.new.values
    tmp = np.where(tmp == '*', 1, 0)
    lcdata.new = tmp
    #tmp = lcdata.sparse.values
    #tmp = np.where(tmp == 'Y', 1, 0)
    #lcdata.sparse = tmp
    #tmp = lcdata.widefield.valuesCometGPs/data/LCLIST_PUB_2019JAN/LC_SUM_PUB.TXT")#
    #tmp = np.where(tmp == 'Y', 1, 0)
    #lcdata.widefield = tmp
    return lcdata

def log_period_prior_kde(lcdb_file = "LCLIST_PUB_CURRENT/LC_SUM_PUB.TXT"):
    """
	Calculates the log period prior distribution based on the
	current asteroid rotational periods documented.
    """

    from os.path import expanduser
    userhome = expanduser("~")

    file_loc = userhome + "/asterogap/data/"


    if not os.path.isfile(file_loc + lcdb_file):
		# if this command doesn't work, you can always got to
		# http://www.minorplanet.info/lightcurvedatabase.html
		# download and unzip the latest public release file
		# and then place that file into the data folder of this package
        print("%s is not found." %lcdb_file)
        print("Downloading the lastest public release from http://www.minorplanet.info/lightcurvedatabase.html.")

        import requests
        import zipfile

        zip_url = "http://www.minorplanet.info/datazips/LCLIST_PUB_CURRENT.zip"
        r = requests.get(zip_url)
        print("test")

		# send a HTTP request to the server and save
		# the HTTP response in a response object called r
        with open("LCLIST_PUB_CURRENT.zip",'wb') as f:
			# write the contents of the response (r.content)
			# to a new file in binary mode.
			# should take a few seconds
            f.write(r.content)

        with zipfile.ZipFile("LCLIST_PUB_CURRENT.zip", 'r') as zip_ref:
            zip_ref.extractall("../data/LCLIST_PUB_CURRENT")

		# go ahead and delete the zip file
        if os.path.isfile(lcdb_file):
            os.remove("LCLIST_PUB_CURRENT.zip")

    lcdb = read_lcdb(file_loc + lcdb_file)
	# drop any entries that have NAN for a period value
    x = np.log(lcdb.period.dropna().values/24.)

    kde = scipy.stats.gaussian_kde(x, bw_method=0.2 / x.std(ddof=1))

    return kde
