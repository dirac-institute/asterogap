=========
AsteroGaP
=========

.. image:: https://img.shields.io/pypi/v/asterogap.svg
        :target: https://pypi.python.org/pypi/asterogap


AsteroGaP (Asteroid Gaussian Processes) is a Bayesian-based Gaussian Process model that seeks to fit sparsely-sampled asteroid light curves.

By utilizing a more flexible Gaussian Process framework for modeling asteroid light curves, we are able to represent light curves in a periodic but non-sinusoidal manner.

* Free software: 3-clause BSD license
* Documentation: https://dirac-institute.github.io/asterogap.
* Accompanying paper published in AJ: https://ui.adsabs.harvard.edu/abs/2022AJ....163...29L/abstract

Installation
------------

At the command line (via pip):

``$ pip install asterogap``


At the command line (via GitHub):

``$ git clone https://github.com/dirac-institute/asterogap.git``

``$ cd asterogap``

``$ python setup.py install``


Citation
--------

If you use AsteroGaP in your work, please cite the accompanying `paper <https://ui.adsabs.harvard.edu/abs/2022AJ....163...29L/abstract>`_. 

Here's the bibtex entry for your convenience:

::

    @ARTICLE{2022AJ....163...29L,
           author = {{Lindberg}, Christina Willecke and {Huppenkothen}, Daniela and {Jones}, R. Lynne and {Bolin}, Bryce T. and {Juri{\'c}}, Mario and {Golkhou}, V. Zach and {Bellm}, Eric C. and {Drake}, Andrew J. and {Graham}, Matthew J. and {Laher}, Russ R. and {Mahabal}, Ashish A. and {Masci}, Frank J. and {Riddle}, Reed and {Shin}, Kyung Min},
            title = "{Characterizing Sparse Asteroid Light Curves with Gaussian Processes}",
          journal = {\aj},
         keywords = {72, 1930, 1900, 1916, 1955},
             year = 2022,
            month = jan,
           volume = {163},
           number = {1},
              eid = {29},
            pages = {29},
              doi = {10.3847/1538-3881/ac3079},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2022AJ....163...29L},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    


