from setuptools import setup

PACKAGE_NAME = "sotabench"
LICENSE = "Apache 2.0"
AUTHOR = "rstojnic"
EMAIL = "hello@sotabench.com"
URL = "https://sotabench.com"
DESCRIPTION = "Easily benchmark Machine Learning models on selected tasks and datasets"


setup(
    name=PACKAGE_NAME,
    maintainer=AUTHOR,
    version='0.0.4',
    packages=[PACKAGE_NAME, 'sotabench.core'],
    include_package_data=True,
    license=LICENSE,
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    url=URL,
    install_requires=[],
)
