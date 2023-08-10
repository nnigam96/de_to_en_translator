from setuptools import setup, find_packages

#with open('requirements.txt') as f:
    #requirements = f.read().splitlines()

setup(
    name='infer_package',
    version='0.1',
    packages=find_packages(),
    install_requires=  ['requests'],
    description='Mini - German to English Translation System using Transformers'
)