from setuptools import setup

PACKAGE_NAME = "sotabench"
LICENSE = "Apache 2.0"
AUTHOR = "Atlas ML"
EMAIL = "ross@atlasml.io"
URL = "https://www.atlas.ml"
DESCRIPTION = "Benchmarking open source deep learning models"


setup(
    name=PACKAGE_NAME,
    maintainer=AUTHOR,
    version='0.001',
    packages=[PACKAGE_NAME,
              'sotabench.core',
              'sotabench.datasets',
              'sotabench.image_classification',
              'sotabench.object_detection',
              'sotabench.semantic_segmentation'],
    include_package_data=True,
    license=LICENSE,
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    url=URL,
    install_requires=['torch', 'torchvision'],
)