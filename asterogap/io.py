import pandas as pd

def read_data(filename, datadir="./", col=(0, 1, 2), whitespace=False):
    """
    Read in light curve data from asteroid.
    """   

    if isinstance(filename, str) is False:
       raise ValueError("filename must be a string with the file name.")

    if isinstance(datadir, str) is False:
       raise ValueError("datadir must be a string with the data directory.")
         
    data = pd.read_csv(datadir + filename, header=None, delim_whitespace=whitespace)
         
    tsample = data[col[0]].to_numpy()
    fsample = data[col[1]].to_numpy()
    flux_err = data[col[2]].to_numpy()
      
    return tsample, fsample, flux_err
