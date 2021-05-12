from os import path, remove
from setuptools import setup, find_packages
import sys
import versioneer


# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (3, 6)
if sys.version_info < min_version:
    error = """
asterogap does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(*(sys.version_info[:2] + min_version))
    sys.exit(error)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]

# download LCDB latest file
lcdb_file = "./data/LCLIST_PUB_CURRENT/LC_SUM_PUB.TXT"

if not path.isfile(lcdb_file):
		# if this command doesn't work, you can always got to
		# http://www.minorplanet.info/lightcurvedatabase.html
		# download and unzip the latest public release file
		# and then place that file into the data folder of this package
        print("%s is not found." %lcdb_file)
        zip_url = "http://www.minorplanet.info/datazips/LCLIST_PUB_CURRENT.zip"
        print("Downloading the lastest public release from %s" %zip_url)

        import requests
        import zipfile

        r = requests.get(zip_url)

		# send a HTTP request to the server and save 
		# the HTTP response in a response object called r 
        with open("LCLIST_PUB_CURRENT.zip",'wb') as f: 
			# write the contents of the response (r.content) 
			# to a new file in binary mode. 
			# should take a few seconds
            f.write(r.content) 

        with zipfile.ZipFile("LCLIST_PUB_CURRENT.zip", 'r') as zip_ref:
            zip_ref.extractall("./data/LCLIST_PUB_CURRENT")

		# go ahead and delete the zip file
        if path.isfile(lcdb_file):
            remove("LCLIST_PUB_CURRENT.zip")

# hopefully this worked



setup(
    name='asterogap',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="AsteroGaP (Asteroid Gaussian Processes) is a Bayesian-based Gaussian Process model that seeks to fit sparsely-sampled asteroid light curves.",
    long_description=readme,
    author="Christina Willecke Lindberg",
    author_email='clindbe2@jhu.edu',
    url='https://github.com/dirac-institute/asterogap',
    python_requires='>={}'.format('.'.join(str(n) for n in min_version)),
    packages=find_packages(exclude=['docs', 'tests']),
    entry_points={
        'console_scripts': [
            # 'command = some.module:some_function',
        ],
    },
    include_package_data=True,
    package_data={
        'asterogap': [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
        ]
    },
    install_requires=requirements,
    license="BSD (3-clause)",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)
