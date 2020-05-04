import pytest
import numpy as np
import scipy.stats
import george

from ..GP import GPFit

class Test_GPFit(object):
    @classmethod
    def setup_class(cls):
        # setting up some input parameters
        time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        flux = np.array([4, 5, 6, 7, 8, 9, 8, 7, 6, 5])
        flux_err = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        kernel_long = True

        cls.asteroid = GPFit(time, flux, flux_err, kernel_long)

    def test_init(self):

        assert len(self.asteroid.time) == 10
        assert len(self.asteroid.time) == len(self.asteroid.flux) == len(self.asteroid.flux_err)

        time_true = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert np.all(self.asteroid.time == time_true)
        # NOTE: np.all only works on np.arrays when comparing to one number

        flux_true = np.array([4, 5, 6, 7, 8, 9, 8, 7, 6, 5])
        assert np.all(self.asteroid.flux == flux_true)
        assert np.all(self.asteroid.flux_err == 1)
        #assert self.asteroid.kernel_long is True


    def test_init_w_incorrect_inputs(self):
        """
        Test that the method shouldn't be able to initialize properly
        if the input parameters aren't correct.
        """

        # Daniela said that setting up the asteroid again shouldn't be a problem
        # setting up some incorrect input parameters (missing the last value)
        time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        flux = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 6])
        flux_err = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        kernel_long = True

        asteroid = GPFit(time, flux, flux_err, kernel_long)

        # AssertionError
        with pytest.raises(AssertionError):
            assert len(asteroid.time) == 10, "Length of time array does not match expectations."
            assert len(asteroid.time) == len(asteroid.flux) == len(asteroid.flux_err), "Inputs are not"
            " the same size. Make sure your time, flux, and flux errors are all there."
            assert np.all(asteroid.flux == 5), "Flux input does not match expectations."
            assert np.all(asteroid.flux_err == 1), "Flux error input does not not match expectations."

    def test_set_params(self):
        """
        Test to see if the set_param method sets up appropriate parameters based
        off the values given.
        """

        # run the method
        self.asteroid.set_params()

        assert self.asteroid.params["mean"] == 6.5, "Mean flux parameter was not computed correctly."
        assert self.asteroid.params["log_amp_k2"] == np.log(self.asteroid.flux.max() - self.asteroid.flux.min()), "Log amp parameter for kernel 1 was not calculated properly. "
        assert self.asteroid.params["gamma"] == 10, "Gamma parameter was not assigned correctly."
        assert self.asteroid.params["log_period"] == 0, "Log period parameter was not assigned correctly."

        if self.asteroid.kernel_long == True:
            assert self.asteroid.params["log_amp_k1"] == np.log(self.asteroid.flux.max() - self.asteroid.flux.min()),  "Log amp parameter for kernel 2 was not calculated properly. "
            assert self.asteroid.params["log_metric"] == np.log(25), "Log metric parameter was not assigned correctly."

        else:
            # check to see that the attributes don't exist
            assert self.asteroid.params.get("log_amp_k1") is None, "Log amp parameter for long_kernel exists when it shouldn't."
            assert self.asteroid.params.get("log_metric") is None, "Log metric parameter for long_kernel exists when it shouldn't."

    def test_set_walker_param_matrix(self):
        """
        Test to see if the set_walker_param_matrix method sets up an appropriate
        matrix of values.
        """
        seed=0
        nwalkers=10

        self.asteroid.set_walker_param_matrix(nwalkers=nwalkers, seed=seed)

        # recreate the true starting matrix
        p_start = np.array(list(self.asteroid.params.values()))
        p0_true = scipy.stats.multivariate_normal.rvs(
            p_start, size=nwalkers, random_state=seed
        )
        np.random.seed(seed)
        p0_true[:, -1] = np.random.normal(size=nwalkers) * 0.5 + np.log(4 / 24.0)

        assert np.all(self.asteroid.walker_params == p0_true), "Walker matrix was not set up correctly."

    def test_set_gp_kernel(self):
        """
        Test to see if the GP kernel is set up correctly or not.
        """

        # set up the expected GP kernel
        if self.asteroid.kernel_long:
            k1 = np.exp(self.asteroid.params["log_amp_k1"]) * george.kernels.ExpSquaredKernel(
                metric=np.exp(self.asteroid.params["log_metric"])
            )

        k2 = np.exp(self.asteroid.params["log_amp_k2"]) * george.kernels.ExpSine2Kernel(
            gamma=(self.asteroid.params["gamma"]), log_period=self.asteroid.params["log_period"]
        )

        if self.asteroid.kernel_long:
            kernel = k1 * k2
        else:
            kernel = k2

        gp_true = george.GP(kernel, fit_mean=True, mean=self.asteroid.params["mean"])
        gp_true.compute(self.asteroid.time, self.asteroid.flux_err)

        self.asteroid.set_gp_kernel()

        # since you can't directly compare kernels, it's easiest to see if they
        # calculate the same things with the same values
        if self.asteroid.kernel_long:
            param_vector = np.array([ 8.26405235,  2.00959512,  4.19761381,  3.85033111, 11.86755799, -0.9097333 ])
        else:
            param_vector = np.array([ 8.26405235,  3.85033111, 11.86755799, -0.9097333 ])

        gp_test = self.asteroid.gp

        gp_test.set_parameter_vector(param_vector)
        gp_true.set_parameter_vector(param_vector)

        lnlike_test = gp_test.lnlikelihood(np.arange(0,10))
        lnlike_true = gp_true.lnlikelihood(np.arange(0,10))


        assert lnlike_test == lnlike_true, "Kernel was not compiled correctly."

    def test_run_emcee(self):
        """
        Test to see if the MCMC run will produce the same results.
        """

        nwalkers = 10
        niter = 100
        burn_in = 10

        self.asteroid.run_emcee(nwalkers=nwalkers, niter=niter, threads=1, burn_in=burn_in)






        # will this method assign the values for params to the top cls class? Cuz I need it later.






# def test_init():
#     """
#     Test that the class is initialized correctly.
#     """
#
#     # setting up some input parameters
#     time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#     flux = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
#     flux_err = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#     kernel_long = True
#
#     asteroid = GPFit(time, flux, flux_err, kernel_long)
#
#     assert len(asteroid.time) == 10
#     assert len(asteroid.time) == len(asteroid.flux) == len(asteroid.flux_err)
#
#     time_true = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#     assert np.all(asteroid.time == time_true)
#     # NOTE: np.all only works on np.arrays when comparing to one number
#     assert np.all(asteroid.flux == 5)
#     assert np.all(asteroid.flux_err == 1)
#     assert asteroid.kernel_long is True

#
# def test_init_w_incorrect_inputs():
#     """
#     Test that the function shouldn't be able to initialize properly
#     if the input parameters aren't correct.
#     """
#
#     # setting up some incorrect input parameters (missing the last value)
#     time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
#     flux = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 6])
#     flux_err = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#     kernel_long = True
#
#     asteroid = GPFit(time, flux, flux_err, kernel_long)
#
#     # AssertionError
#     with pytest.raises(AssertionError):
#         assert len(asteroid.time) == 10, "Length of time array does not match expectations."
#         assert len(asteroid.time) == len(asteroid.flux) == len(asteroid.flux_err), "Inputs are not"
#         " the same size. Make sure your time, flux, and flux errors are all there."
#         assert np.all(asteroid.flux == 5), "Flux input does not match expectations."
#         assert np.all(asteroid.flux_err == 1), "Flux error input does not not match expectations."
    # time_true = [0,1,2,3,4,5,6,7,8,9]
    # assert np.all(asteroid.time == time_true)
    # NOTE: np.all only works on np.arrays when comparing to one number
    #
    #
    # assert kernel_long == True


def test():
    """

    """
