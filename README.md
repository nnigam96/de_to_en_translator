% MINI GERMAN-TO_ENGLISH TRANSLATOR

This repository contains the code for a Simple German to English Translation system using Tranformer models.

The repository is organized into 3 main folders:
model_package: Contains the code for the model and the training script
infer_package: Contains the code for the inference script
demo_api: Contains the code for the demo API

1. #model_package:
    a. model_package: Contains .py files used for training and saving the neural network 
        -driver.py: Main entry point for the training routine
        -model.py: Contains the code for the model
        -data.py: Contains the code for data loaders and data transforms
        -data_utils: Utility functions for data processing
        -trainer: Contains the code for the training routine
        -constants: Contains the constants used in the model
        -utils: Utility functions for the model training
        -inference: Contains the code for the inference routine, used by Flask API in demo_api
    b. tests: Placeholder for the unit tests for the model
    c. requirements.txt: Contains the dependencies for the model
    d. setup.py: Contains the setup script for the package

2. #infer_package:
    a. ##infer_package:
        - translate.py: Contains the code for the inference routine which is used by the API for exposing the end point
    b. ##tests: Placeholder for the unit tests for the inference script
    c. ##setup.py: Contains the setup script for the package

3. #demo_api:
    a. demo_api:
        - app.py: Contains the code for the Flask API with the end point for the translation


#Important commands for usage:
1. ##To train the model:
    - Navigate to model_package/model_package
    - Run the command: 
    ```
    python driver.py
    ```
    - The model will be saved in the model_package/model_package/checkpoints folder
    - Model hyperparameters can be updated in the model_package/model_package/constants.py file

2. ##To run the flask app:
    - Navigate to demo_api/demo_api
    - Run the command: 
    ```
    python app.py
    ```
    - The app will be running locally on port 5000

#Package Creation:
1. ##To create the model package:
    - Navigate to model_package
    - Run the command: 
    ```
    python setup.py sdist bdist_wheel
    ```
    - The package will be created in the dist folder

2. ##To create the inference package:
    - Navigate to infer_package
    - Run the command: 
    ```
    python setup.py sdist bdist_wheel
    ```
    - The package will be created in the dist folder

#Install Package after creation:
1. To install the model package:
    ```
    # Navigate to repo directory
    cd model_package
    pip install dist\model_package-0.1.tar.gz
    ```
2. To install the inference package:
    ```
    # Navigate to repo directory
    cd infer_package
    pip install dist\infer_package-0.1.tar.gz
    ```


NOTE: For infer_package to run, the system expects an installation of 2 components, one each for German and English languages.
The installation for the same can be found here. Ignore if installed already:

```
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm

```

These are used internally for the tokenization of the text.

Once sucessfully installed, infer_package can be used for translation as follows:

```
import infer_package.translate as translator
translator.get_translation("Ich bin ein Berliner")
>>> 'I am a Berliner'
```