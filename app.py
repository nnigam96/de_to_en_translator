import json

from constants import *
from model import *
from data import *
from data_utils import *
from trainer import *
from torch.utils.data import DataLoader
from utils import *
from inference import *
import argparse
import torch
from flask import Flask, jsonify, request, render_template


app = Flask(__name__)
token_transform, vocab_transform = get_vocab_transforms()

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
model.load_state_dict(torch.load(r"D:\Git Repos\modlee_code_test\checkpoints\best_model.pth"))
model = model.to(DEVICE)


#@app.route('/', methods=['GET', 'POST'])
#def index():
    #return render_template('index.html')
@app.route('/translate', methods=['POST'])
def translate_text():
    if request.method == 'POST':
        input_data = request.form.get('input_text')
        
        if input_data:
            translation = translate(model, input_data)
            result = {'input':input_data, 'translation': translation}
            return jsonify(result)
        else:
            return jsonify({'error': 'Input text not provided'}), 400

@app.route('/fetch-modelparameters', methods=['GET'])
def fetch_hyperparameters():
    if request.method == 'GET':
        result = {'num_encoder_layers': NUM_ENCODER_LAYERS, 
                  'num_decoder_layers': NUM_DECODER_LAYERS, 
                  'embedding_size': EMB_SIZE, 
                  'num_heads': NHEAD, 
                  'src_vocab_size': SRC_VOCAB_SIZE, 
                  'tgt_vocab_size': TGT_VOCAB_SIZE, 
                  'ffn_hidden_dim': FFN_HID_DIM,
                  'learning_rate': LEARNING_RATE,
                  'batch_size': BATCH_SIZE,
                  'num_epochs': NUM_EPOCHS
                  }
        return jsonify(result)

@app.route('/fetch-model', methods=['GET'])
def fetch_model():
    if request.method == 'GET':
        result = {'model': str(model), 
                  
                  }
        return jsonify(result)

#@app.route('/model_arch', methods=['GET'])
#def fetch_model_arch():


if __name__ == '__main__':
    app.run(port=5000)