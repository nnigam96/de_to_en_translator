import requests

def get_translation(text):
    """Gets the translation of a German text to English.

    Args:
        text (str): The German text to translate.

    Returns:
        str: The English translation of the text.
    """

    url = "http://localhost:5000/translate"
    data = {"input_text": text} #, "source_language": "de", "target_language": "en"}
    response = requests.post(url, data=data)

    if response.status_code == 200:
        return response.json()["translation"]
    else:
        raise Exception("Error translating text: {}".format(response.status_code))

def get_model_hyperparameters():
    """Gets the list of supported languages.

    Returns:
        JSON -  key: Hyperparameter, value: Value.
    """

    url = "http://localhost:5000/fetch-modelparameters"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Error getting model hyperparameters: {}".format(response.status_code))
    
def get_model_structure():
    """Gets the model structure used for translation.

    Returns:
        JSON of nn.Module structure used for translation.
    """

    url = "http://localhost:5000/fetch-model"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Error getting model hyperparameters: {}".format(response.status_code))
    
def get_compatible_languages():
    """Gets the list of supported languages.

    Returns:
        JSON -  key: Language, value: Language code.
    """

    url = "http://localhost:5000/fetch-compatible-lang"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Error getting compatible languages: {}".format(response.status_code))
