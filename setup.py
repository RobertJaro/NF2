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
                      'numpy', 'matplotlib', 'astropy', 'drms', 'wandb>=0.13', 'lightning==1.9.3', 'pytorch_lightning==1.9.3',
                      'pfsspy'],
    entry_points={
        'console_scripts': [
            'nf2-extrapolate = nf2.extrapolate:main',
            'nf2-extrapolate-series = nf2.extrapolate_series:main',
            'nf2-noaa-to-sharp = nf2.data.noaa_to_sharp:main',
            'nf2-download = nf2.data.download_range:main',
            'nf2-to-vtk = nf2.convert.nf2_to_vtk:main',
            'nf2-to-npy = nf2.convert.nf2_to_npy:main',
            'nf2-to-fits = nf2.convert.nf2_to_fits:main',
            'nf2-to-hdf5 = nf2.convert.nf2_to_hdf5:main',
            'nf2-convert-series = nf2.convert.convert_series:main',
        ]}
)
