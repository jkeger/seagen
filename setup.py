import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="seagen",
    packages=setuptools.find_packages(),
    version="1.0",
    description="Stretched Equal Area Generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jacob Kegerreis",
    author_email="jacob.kegerreis@durham.ac.uk",
    url="https://github.com/jkeger/seagen",
    download_url="https://github.com/jkeger/seagen/archive/1.0.tar.gz",
    license="GNU GPL",
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)"
        ],
    install_requires=["numpy", "matplotlib", "sys"],
    python_requires=">=2",
    keywords=["particle arrangement density SPH sphere shell"],
)