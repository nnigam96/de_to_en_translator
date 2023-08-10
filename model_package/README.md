Contains the code for training and evaluating a pytorch transformer for a German to English translation task. 

#Important files:

1. #model_package\:
    a. driver.py: Main entry point for the training routine
        - Written so that simply executing the file will start the training process and save the model in the checkpoints folder along with loss curves for later analysis.
        - Hyperparameters can be updated in the constants.py file
    
    b. constants.py: 
        -Contains the constants used in the model. One can change  value in this file to change the hyperparameters of the model. before triggering the training process.
    
    c. model.py: Contains the code for the model built using pyTorch
    
    d. data.py: Contains the code for data loaders and data transforms
    
    e. data_utils: Utility functions for data processing
    
    f. trainer: Contains the code for the training routine
    
    g. utils: Utility functions for the model training


2. #setup.py: Contains the setup script for the package
    - Contains the script to create the package. The file is used to create the package using the command mentioned above. Dependencies for the package can be added here for easier installation.