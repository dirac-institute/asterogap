import pytest
import numpy as np

from ..run_gp import read_data


def test_read_data():

    filename = "test_data.csv"
    datadir = "data/test_data/"

    read_data(filename, datadir)


def test_outputs():
    filename = "test_data.csv"
    datadir = "data/test_data/"

    time, flux, flux_err = read_data(filename, datadir)

    assert len(time) == 10
    assert len(time) == len(flux) == len(flux_err)

    time_true = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert np.all(time == time_true)
    assert np.all(flux == 5)
    assert np.all(flux_err == 1)


def test_read_data_fails_with_whitespace_data_in_wrong_format():
    filename = "test_data.csv"
    datadir = "data/test_data/"

    # figure out what error it fails with
    # for this case it's a KEY ERROR
    with pytest.raises(KeyError):
        read_data(filename, datadir, whitespace=True)


has_george = False


@pytest.mark.skipif("not has_george")
def test_george_works():
    pass
