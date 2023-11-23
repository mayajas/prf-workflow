# setup.py
from setuptools import setup, find_packages

setup(
    name='prf-package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open('requirements.txt') if line.strip()
    ],
)