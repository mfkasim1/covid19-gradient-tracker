import os
from setuptools import setup, find_packages

setup(
    name='covidtracker',
    version="0.0.1",
    description='COVID-19 gradient tracker',
    url='https://github.com/mfkasim91/covid19-gradient-tracker',
    author='mfkasim91',
    author_email='firman.kasim@gmail.com',
    license='MIT',
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.8.2",
        "scipy>=0.15",
        "matplotlib>=1.5.3",
        "torch>=1.3",
        "pyro-ppl>=1.3.0",
        "Jinja2>=1.1.0",
        "geopandas>=0.7.0",
        "tqdm>=4.43.0"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",

        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="project library linear-algebra autograd",
    zip_safe=False
)
