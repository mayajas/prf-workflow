from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip()]

setup(
    name='prf-workflow',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
)