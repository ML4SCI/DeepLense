from pathlib import Path

from setuptools import find_packages, setup

# Package meta-data.
NAME = "deeplense-transformers"
DESCRIPTION = "DeepLense library for classiying Dark Matter Halos using transformers"
URL = "https://github.com/sachdevkartik/DeepLense/tree/PR_kartik_transformers"
EMAIL = "kartik,sachdev25@gmail.com"
AUTHOR = "Kartik Sachdev"
REQUIRES_PYTHON = ">=3.6.0"

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Load the package's VERSION file as a dictionary.
about = {}
ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR
PACKAGE_DIR = ROOT_DIR
with open(PACKAGE_DIR) as f:
    _version = f.read().strip()
    about["__version__"] = _version


# What packages are required for this module to be executed?
def list_reqs(fname="requirements.txt"):
    with open(REQUIREMENTS_DIR / fname) as fd:
        return fd.read().splitlines()


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(include=["models", "utils"]),
    package_data={"deeplense": ["VERSION"]},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license="",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        # "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords=[
        "Transformers",
        "Gravitational Lensing",
        "Image Classification",
        "Dark Matter",
    ],
)
