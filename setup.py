import setuptools
from setuptools import setup

setup(
    name='nf2',
    version='v0.3',
    packages=setuptools.find_packages(),
    url='https://github.com/RobertJaro/NF2',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Robert Jarolim',
    author_email='',
    description='Neural Network Force Free magnetic field extrapolation',
    install_requires=['torch>=1.8', 'sunpy[all]>=3.0', 'scikit-image', 'scikit-learn', 'tqdm',
                      'numpy', 'matplotlib', 'astropy', 'drms', 'wandb>=0.13', 'lightning==1.9.3', 'pytorch_lightning==1.9.3'],
)
