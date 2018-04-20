from distutils.core import setup
setup(
    name='seagen',
    packages=['seagen'],
    version='0.1',
    description='Stretched Equal Area Generator',
    long_description=(
        'A python implementation of the stretched equal area (SEA) algorithm '
        'for generating spherically symmetric arrangements of particles with '
        'accurate particle densities, as presented in '
        'Kegerreis et al. (2018), in prep.'
        ),
    author='Jacob Kegerreis',
    author_email='jacob.kegerreis@durham.ac.uk',
    url='https://github.com/jkeger/seagen',
    download_url='https://github.com/jkeger/seagen/archive/0.1.tar.gz',
    license='GNU GPL',
    classifiers=[
        'Programming Language :: Python :: 3',
        ],
    install_requires=[
        'matplotlib', 'scipy', 'numpy', 'numba', 'typing', 'tqdm'
        ],
    python_requires='>=3',
    keywords=['particle', 'arrangement', 'density', 'SPH', 'sphere', 'shell'],
)
