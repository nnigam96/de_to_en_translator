from setuptools import setup, find_packages

#with open('requirements.txt') as f:
    #requirements = f.read().splitlines()

setup(
    name='model_package',
    version='0.1',
    packages=find_packages(),
    install_requires=  [    
                        'Flask==2.3.2',
                        'matplotlib==3.7.2',
                        'nltk==3.8.1',
                        'sacrebleu==2.3.1',
                        'torch==2.0.1',
                        'torchtext==0.15.2',
                        'spacy',
                        ],
    description='Package to access model training files used for developing a Mini - German to English Translation System using Transformers'
)