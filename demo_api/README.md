## This directory contains the code for the demo API.

### app.py: 
     - Contains the code for the Flask API with the end point for the translation

### Methods in app.py:
### 1. translate_text (URL: /translate)
    - Method: POST
    - Input (JSON): 
        - text: Text to be translated
        - source_language: Language of the text to be translated
        - target_language: Language to which the text needs to be translated

### 2. fetch_compatible_lang (URL: /fetch-compatible-lang)
    - Method: GET
    - Input: None
    - Output: List of supported languages

### 3. evaluate_score (URL: /evaluate-score)
    - Method: POST
    - Input: 
        - input_text: Generated translationText to be translated
        - reference_text: Reference translation
        - metric_type: Score to be evaluated, one of BLEU or METEOR
    - Output: Score of the generated translation

### 4. fetch_model (URL: /fetch-model)
    - Method: GET
    - Input: None
    - Output: Model Architecture in a json format (targeted toward developers)

### 5. fetch_hyperparameters (URL: /fetch-modelparameters)
    - Method: GET
    - Input: None
    - Output: Model hyper-parameters in a json format (targeted toward developers)


### API Usage:

Python: Recommened usage is using the infer_package. Refer infer_package/README.md for more details about it.

Other Languages: Once the app is running, the API can be accessed using the following URL:
```
'http://localhost:5000/--URL--'
```
where --URL-- is one of the above mentioned URLs.

### NOTE: 
    For scenarios where package cannot be installed, request can be sent using the httpie package. Refer to the following link for more details: https://httpie.org/doc#installation

### Reference Code:
```
http --form POST http://localhost:5000/translate input_text="Eine Gruppe von Menschen steht vor einem Iglu." 
```
