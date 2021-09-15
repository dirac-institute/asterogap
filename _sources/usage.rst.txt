=====
Usage
=====

Start by making sure that you have AsteroGaP installed.

.. code-block:: python

    import asterogap

If nothing yells at you, then you're good and can proceed. If something does seem upset, go back and reinstall.

---------------
Reading in Data
---------------

The code runs primarily on the command line, where all you really need is a file with columns of time, fluxes (or magnitudes), and flux errors (or magnitude errors). This documentation will go into how you can fully customize your input and how it will effect the run.

By default, the code will assume that the first 3 columns correspond to time, flux, and flux errors respectively and that there is no header. If your file does have a header, you need to specify the columns by including a ``-c`` or ``--columns`` and listing the names of the columns afterwards (again, in the order of time, flux, and flux error respectively). You do not need to put the names in quotations.

``$ python run_gp -f [filename] -d [datadir] -c [time] [flux] [flux error]``

In order to read in the data file correctly, the code needs to know that the delimeter of your data is (e.i. is your data separated by whitespace " " or by commas ","). The code will assume commas by default but if your data is seperated by whitespace, then you can indicate that on the command line with a ``-ws`` or ``--whitespace``.

``$ python run_gp -f [filename] -d [datadir] -ws``

-------------
Prepping MCMC
-------------

The code utilizes MCMC methods to generate results. For more information on MCMC and the package used here, check out `Dan Foreman-Mackey's emcee paper <https://emcee.readthedocs.io/en/stable/>`_.

There are a couple different parameters that will greatly effect how your MCMC code will run.

* Walkers/chains (default: 100)
* Burn-in iterations (default: 2000): the number of steps the MCMC should run through in order to ensure that the chains have converged properly (or the number of iterations to remove from the head of the MCMC chains).
* Iterations (default: 1000)

``$ python run_gp -f [filename] -d [datadir] -w [nwalkers] -b [burn_in] -i [niter]``

All of these effect how long it will take for your code to run, with more steps or walkers taking more time.

If you want to multithread the run, you can set the number of threads for the emcee code to use. The default is 1. This is great if you have access to high-performance resources, but if you're working on a personal computer, just make sure not to set the thread number to more than what your machine can handle.

``$ python run_gp -f [filename] -d [datadir] -w [nwalkers] -b [burn_in] -i [niter] -t [threads]``

------------------
Setting the kernel
------------------

For the Gaussian Process kernel, the priors are pre-encoded, but there is the option of setting up a long-term lightcurve profile adjustment kernel. This additional kernel is supposed to account for the gradual change in the lightcurve profile that is naturally seen in most asteroids. Without it, the kernel remains strictly periodic with no fine adjustments to the amplitudes.

This kernel is by default included in the setup, but you can omit it by indicating it on the command line with ``-k`` or ``--kernel``.

``$ python run_gp -f [filename] -d [datadir] -k``



.. autofunction:: asterogap.run_gp.read_data

.. autofunction:: asterogap.run_gp.write_data


The main powerhouse for this code is the Gaussian Process (GP) class, called GP Fit. With this class, we store all the information regarding the input data, the parameters, the kernel settings, and the results from the MCMC runs.

.. autoclass:: asterogap.GP.GPFit
	:members:

.. autofunction:: asterogap.GP.prior

.. autofunction:: asterogap.GP.post_lnlikelihood
