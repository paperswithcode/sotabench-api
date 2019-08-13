import io
from setuptools import setup, find_packages
from sotabenchapi.version import __version__

name = "sotabenchapi"
author = "Robert Stojnic"
author_email = "hello@sotabench.com"
license = "Apache-2.0"
url = "https://sotabench.com"
description = (
    "Easily benchmark Machine Learning models on selected tasks and datasets."
)


setup(
    name=name,
    version=__version__,
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    description=description,
    long_description=io.open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url=url,
    platforms=["Windows", "POSIX", "MacOSX"],
    license=license,
    packages=find_packages(),
    install_requires=io.open("requirements.txt").read().splitlines(),
    entry_points="""
        [console_scripts]
        sb=sotabenchapi.__main__:cli
    """,
)
