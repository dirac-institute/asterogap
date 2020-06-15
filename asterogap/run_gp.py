import argparse
import textwrap

import pandas as pd
import h5py

# from plotting import plot_mcmc_sampling_results
from asterogap.GP import GPFit

import os

# TODO: is this still needed?
os.environ["MKL_NUM_THREADS"] = "3"


def read_data(filename, datadir="./", cols=None, whitespace=False):
    """
    Read in light curve data from asteroid.
    """

    if cols is None:
        header = None
        cols = [0, 1, 2]

    else:
        header = 0

    data = pd.read_csv(datadir + filename, delim_whitespace=whitespace, header=header)

    print("columns = " + str(cols))

    tsample = data[cols[0]]
    fsample = data[cols[1]]
    flux_err = data[cols[2]]

    return tsample, fsample, flux_err


def write_data(filename, sampler, asteroid, nwalkers, niter, burn_in):
    """
    Write the sampler results as an HDF5 file,
    with all the other info you might want.
    """

    # create a new filename ending
    filename_new = filename.replace(filename.split(".")[-1], "hdf5")

    with h5py.File(filename_new, "w") as f:
        f.create_dataset("chain", data=sampler.chain)

        f.attrs["walkers"] = nwalkers
        f.attrs["iterations"] = niter
        f.attrs["data_pts"] = asteroid.data_pts
        f.attrs["acceptance_fraction"] = sampler.acceptance_fraction
        f.attrs["burn_in"] = burn_in
        f.create_dataset("time", data=asteroid.time)
        f.create_dataset("flux", data=asteroid.flux)
        f.create_dataset("flux_err", data=asteroid.flux_err)


def main():
    # read in the data file
    print("\nreading in data")
    time, flux, flux_err = read_data(filename, datadir, cols, whitespace)

    if kernel_long:
        print("\nincluding long-term kernel\nit's log unif!")
    asteroid = GPFit(time, flux, flux_err, kernel_long)

    print("\nsetting kernel")
    asteroid.set_params()
    asteroid.set_walker_param_matrix(nwalkers)
    asteroid.set_gp_kernel()

    print("\nrunning mcmc")
    sampler = asteroid.run_emcee(
        niter=niter, nwalkers=nwalkers, burn_in=burn_in, threads=threads
    )

    print("\nwriting out results")
    write_data(filename, sampler, asteroid, nwalkers, niter, burn_in)

    return


if __name__ == "__main__":
    # DEFINE PARSER FOR COMMAND LINE ARGUMENTS
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=" ",  # Bayesian QPO searches for burst light curves.",
        epilog=textwrap.dedent(
            """

    NOTE! The first 3 columns of your input file "-f" must correspond to your
    time, flux, and flux error in that order. All columns beyond column 3 will be ignored.

    Examples
    --------

    Print this help message:

            $> python run_gp.py --help

    Run this script from anywhere on your system:

            $> python /absolute/path/to/asterogap/run_gp.py --help


    # Run on example data in the data directory:
    #
    #         $> python /absolute/path/to/asterogap/run_gp.py -f "2001SC170.csv"
    #                 -d "absolute/path/to/CometGP/data/asteroid_csv"

    Run on example data (from example data directory) with more walkers, steps, etc.

            $> python ../code/run_gp.py -f "2001SC170.csv" -d "./" -w 50 -i 5000 -t 2


    """
        ),
    )

    parser.add_argument(
        "-f",
        "--filename",
        action="store",
        dest="filename",
        required=True,
        help="Data file with observed time (in unit days) and flux.",
    )
    parser.add_argument(
        "-d",
        "--datadir",
        action="store",
        dest="datadir",
        required=False,
        default="./",
        help="Directory with the data (default: current directory).",
    )
    parser.add_argument(
        "-w",
        "--nwalkers",
        action="store",
        dest="nwalkers",
        required=False,
        type=int,
        default=100,
        help="The number of walkers/chains for the MCMC run (default: 100).",
    )
    parser.add_argument(
        "-i",
        "--niter",
        action="store",
        dest="niter",
        required=False,
        type=int,
        default=1000,
        help="The number of iterations per chain/walker in the MCMC run (default: 1000).",
    )
    parser.add_argument(
        "-t",
        "--threads",
        action="store",
        dest="threads",
        required=False,
        type=int,
        default=1,
        help="The number of threads used for computing the posterior (default: 1).",
    )

    parser.add_argument(
        "-ws",
        "--whitespace",
        action="store_true",
        dest="whitespace",
        required=False,
        default=False,
        help="The delimeter for the input file, assumed not to be whitespace.",
    )

    parser.add_argument(
        "-b",
        "--burn_in",
        action="store",
        dest="burn_in",
        required=False,
        type=int,
        default=2000,
        help="The number of iterations to remove from the head of the MCMC chain walkers.",
    )
    parser.add_argument(
        "-k",
        "--kernel",
        action="store_false",
        dest="kernel",
        required=False,
        default=True,
        help="Include a long-term lightcurve profile adjustment kernel.",
    )
    parser.add_argument(
        "-c",
        "--columns",
        nargs=3,
        type=str,
        action="store",
        dest="columns",
        required=False,
        help="Specify which column names to use to extract time, flux, and flux error. Must be a string.",
    )

    clargs = parser.parse_args()

    filename = clargs.filename
    datadir = clargs.datadir
    nwalkers = clargs.nwalkers
    niter = clargs.niter
    threads = clargs.threads
    whitespace = clargs.whitespace
    burn_in = clargs.burn_in
    kernel_long = clargs.kernel
    cols = clargs.columns

    main()
