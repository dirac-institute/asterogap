import pytest
import numpy as np

try:
   import george
   has_george = True
except ImportError:
   has_george = False

from ..io import read_data

def test_read_data_runs():
    filename = "test_data.csv"
    datadir = "../../data/"

    read_data(filename, datadir)

def test_read_data_fails_with_nonsense_inputs():
    with pytest.raises("ValueError"):
        read_data(0, 1)

def test_outputs():
    filename = "test_data.csv"
    datadir = "../../data/"

    time, flux, fluxerr = read_data(filename, datadir)
    assert len(time) == 10, "Time should be of length 10"
    assert len(time) == len(flux) == len(fluxerr)
    
    time_true = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert np.all(time == time_true)
    assert np.all(flux == 5)
    assert np.all(fluxerr == 1)

def test_read_data_works_with_tsv():
    filename = "test_data.tsv"
    datadir = "../../data/"

    read_data(filename, datadir, whitespace=True)

def test_read_data_fails_with_wrong_format():
    filename = "test_data.csv"
    datadir = "../../data/"

    with pytest.raises(KeyError):
        time, flux, fluxerr = read_data(filename, datadir, whitespace=True)

def test_read_data_failes_with_whitespace_data_in_wrong_format():
    filename = "test_data.tsv"
    datadir = "../../data/"

    with pytest.raises(KeyError):
        time, flux, fluxerr = read_data(filename, datadir, whitespace=False)

def test_cols_works_correctly():

    filename = "test_data_columns.csv"
    datadir = "../../data/"
    cols = (1, 4, 6)

    time, flux, fluxerr = read_data(filename, datadir, col=cols, whitespace=False)
    time_true = [1, 2, 3, 4, 5]
    flux_true = 5 
    fluxerr_true = 1
    assert np.all(time == time_true)
    assert np.all(flux == flux_true)
    assert np.all(fluxerr == fluxerr_true) 

@pytest.mark.skipif("not has_george")
def test_george_works():
    # do something
    pass


