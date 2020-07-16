import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="seagen",
    packages=setuptools.find_packages(),
    version="1.4.1",
    description="Stretched Equal Area Generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jacob Kegerreis",
    author_email="jacob.kegerreis@durham.ac.uk",
    url="https://github.com/jkeger/seagen",
    project_urls={
        "Paper": "https://doi.org/10.1093/mnras/stz1606",
        "Group": "http://icc.dur.ac.uk/giant_impacts",
        "WoMa": "https://github.com/srbonilla/WoMa",
    },
    license="GNU GPLv3+",
    py_modules=["seagen"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Development Status :: 4 - Beta",
        ],
    python_requires=">=3",
    install_requires=["numpy"],
    keywords=["particle arrangement density sphere shell SPH initial conditions"],
)
