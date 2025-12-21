"""Project Setup"""

from setuptools import find_packages, setup

PACKAGE_VERSION = "0.1.0"

requirements = [
    "numpy==2.2.6",
    "matplotlib>=3.7.2",
    "seaborn==0.11.2"
]

setup(
    name="ml-codes",   # Package name
    version=PACKAGE_VERSION,  # Set package version manually
    packages=find_packages(where=".", exclude=["tests", "tests.*"]),
    install_requires=requirements,
    python_requires="==3.10.6",  # Ensuring only Python 3.10.6 is used
    keywords="ml-codes",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
)
