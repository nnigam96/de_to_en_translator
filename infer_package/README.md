## The directory contains the code used in the inference package. The package can be created and installed using the following command:

```
# Navigate to repo directory
cd infer_package
python setup.py sdist bdist_wheel

# Install the package
pip install dist\infer_package-0.1.tar.gz
```

## The package can be used in the following way:

```
fimport infer_package.translate as translator
translator.get_translation("Ich bin ein Berliner")
>>> 'I am a Berliner'
```

The purpose of the package is to provide a wrapper around the api to make it easier to use. The package contains 2 files:

1. infer_package\translate.py: 
    - Contains the code for api interaction. It uses the request library in python to send and fetch request to the Flask API. 
    - *IMP* The code needs to be updated once a new endpoint is added to the api in the demp_api/app.py file, to access to that via the package
    - Please ensure to repackage the files after any changes are made to the code. You can refer to the code from above to create the package.

2. setup.py:
    - Contains the script to create the package. The file is used to create the package using the command mentioned above. Dependencies for the package can be added here for easier installation. 
