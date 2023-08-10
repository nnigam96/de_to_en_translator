from model_package.constants import *
from model_package.inference import *
from flask import Flask, jsonify, request

# Create a Flask app
app = Flask(__name__)

api_ref_dict = fetch_model()

# Endpoint to translate input text
@app.route('/translate', methods=['POST'])
def translate_text():
    if request.method == 'POST':
        input_data = request.form.get('input_text')
        
        if input_data:
            translation = translate(api_ref_dict['model'], input_data)
            result = {'input': input_data, 'translation': translation}
            return jsonify(result)
        else:
            return jsonify({'error': 'Input text not provided'}), 400

# Endpoint to fetch hyperparameters and model configuration
@app.route('/fetch-modelparameters', methods=['GET'])
def fetch_hyperparameters():
    if request.method == 'GET':
        result = {'num_encoder_layers': NUM_ENCODER_LAYERS, 
                  'num_decoder_layers': NUM_DECODER_LAYERS, 
                  'embedding_size': EMB_SIZE, 
                  'num_heads': NHEAD, 
                  'src_vocab_size': api_ref_dict['src_vocab_size'], 
                  'tgt_vocab_size': api_ref_dict['tgt_vocab_size'], 
                  'ffn_hidden_dim': FFN_HID_DIM,
                  'learning_rate': LEARNING_RATE,
                  'batch_size': BATCH_SIZE,
                  'num_epochs': NUM_EPOCHS
                  }
        return jsonify(result)

# Endpoint to fetch the model's state representation as a string
@app.route('/fetch-model', methods=['GET'])
def fetch_model():
    if request.method == 'GET':
        result = {'model': str(api_ref_dict['model'])}
        return jsonify(result)

# Endpoint to fetch the model's compatible languages
@app.route('/fetch-compatible-lang', methods=['GET'])
def fetch_compatible_lang():
    if request.method == 'GET':
        result = {'SRC_LANG': SRC_LANGUAGE,
                  'TGT_LANG': TGT_LANGUAGE}
        return jsonify(result)


# Endpoint to get evaluation metrics for a given input text
@app.route('/evaluate-score', methods=['POST'])
def evaluate_score():
    if request.method == 'POST':
        input_data = request.form.get('input_text')
        refernce_text = request.form.get('reference_text')
        metric_type = request.form.get('metric_type')
        if input_data is None:
            return jsonify({'error': 'Input text not provided'}), 400
        elif refernce_text is None:
            return jsonify({'error': 'Reference text not provided'}), 400
        elif metric_type != 'BLEU' and metric_type != 'METEOR':
            return jsonify({'error': 'Metric type not provided, Supported metric types are BLEU and METEOR'}), 400
        else:
            score = metrics(refernce_text, input_data, metric_type)
            result = {'input': input_data, 'reference': refernce_text, 'score': score}
            return jsonify(result)
        
# List of Additional endpoints to be added:
# 1. Endpoint to fetch the model's weights
# 2. Endpoint to fetch the model's optimizer state
# 3. Endpoint to fetch the model's instantiated object for training from scratch
# 4. Endpoint to pass inference inputs to annotators for evaluation
# ....

# Run the Flask app on port 5000
if __name__ == '__main__':
    app.run(port=5000)
