import pytest
import numpy as np

from ..GP import GPFit


def test_init():
    """
    Test that the class is initialized correctly.
    """

    # setting up some input parameters
    time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    flux = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
    flux_err = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    kernel_long = True

    asteroid = GPFit(time, flux, flux_err, kernel_long)

    assert len(asteroid.time) == 10
    assert len(asteroid.time) == len(asteroid.flux) == len(asteroid.flux_err)

    time_true = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert np.all(asteroid.time == time_true)
    # NOTE: np.all only works on np.arrays when comparing to one number
    assert np.all(asteroid.flux == 5)
    assert np.all(asteroid.flux_err == 1)
    assert kernel_long is True


def test_init_w_incorrect_inputs():
    """
    Test that the function shouldn't be able to initialize properly
    if the input parameters aren't correct.
    """

    # setting up some incorrect input parameters (missing the last value)
    time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    flux = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 6])
    flux_err = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    kernel_long = True

    asteroid = GPFit(time, flux, flux_err, kernel_long)

    # AssertionError
    with pytest.raises(AssertionError):
        assert len(asteroid.time) == 10, "Length of time array does not match expectations."
        assert len(asteroid.time) == len(asteroid.flux) == len(asteroid.flux_err), "Inputs are not the same size. Make sure your time, flux, and flux errors are all there."
        assert np.all(asteroid.flux == 5), "Flux input does not match expectations."
        assert np.all(asteroid.flux_err == 1), "Flux error input does not not match expectations."
    # time_true = [0,1,2,3,4,5,6,7,8,9]
    # assert np.all(asteroid.time == time_true)
    # NOTE: np.all only works on np.arrays when comparing to one number
    #
    #
    # assert kernel_long == True


def test():
    """

    """
