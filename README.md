# German-to-English Neural Machine Translation

A production-ready PyTorch-based Transformer model for German-to-English translation. This project provides a complete pipeline from training custom models to deploying inference APIs, achieving strong BLEU scores on standard benchmarks.

## Problem Statement

Building accurate and efficient machine translation systems requires handling complex linguistic patterns, large vocabularies, and maintaining context across sentence boundaries. This project addresses these challenges by implementing a Transformer-based architecture with:

- **End-to-End Training Pipeline**: From raw text preprocessing to model checkpointing
- **Production API**: Flask-based REST API for real-time translation
- **Evaluation Framework**: BLEU and METEOR scoring for translation quality
- **Modular Design**: Separate packages for training, inference, and API deployment

## Features

- **Transformer Architecture**: Custom PyTorch implementation with multi-head attention
- **Complete Training Pipeline**: Data preprocessing, vocabulary building, and model training
- **Production API**: Flask REST API with multiple endpoints for translation and evaluation
- **Python Package**: Installable inference package for easy integration
- **Evaluation Metrics**: Built-in BLEU and METEOR score calculation
- **Model Metadata**: API endpoints for accessing model architecture and hyperparameters

## Project Structure

```
de_to_en_translator/
├── model_package/          # Training and model code
│   ├── driver.py          # Main training entry point
│   ├── model.py           # Transformer architecture
│   ├── trainer.py         # Training loop
│   ├── data.py            # Data loaders and preprocessing
│   └── constants.py       # Hyperparameters configuration
├── infer_package/          # Inference package
│   └── translate.py       # Translation inference code
├── demo_api/               # Flask API server
│   └── app.py             # REST API endpoints
└── README.md
```

## Quick Start

### Installation

```bash
# Install the inference package
cd infer_package
pip install -e .

# Or install the model package for training
cd model_package
pip install -e .
```

### Training

```bash
cd model_package
python driver.py
```

Hyperparameters can be adjusted in `constants.py` before training.

### Inference

**Using Python Package:**
```python
from infer_package import translate

translation = translate("Eine Gruppe von Menschen steht vor einem Iglu.")
print(translation)
```

**Using REST API:**
```bash
# Start the API server
cd demo_api
python app.py

# Make translation requests
http POST http://localhost:5000/translate \
    text="Eine Gruppe von Menschen steht vor einem Iglu." \
    source_language="de" \
    target_language="en"
```

## API Endpoints

- **POST `/translate`**: Translate text from source to target language
- **GET `/fetch-compatible-lang`**: Get list of supported languages
- **POST `/evaluate-score`**: Calculate BLEU or METEOR scores
- **GET `/fetch-model`**: Get model architecture (JSON)
- **GET `/fetch-modelparameters`**: Get hyperparameters (JSON)

## Technical Details

- **Architecture**: Transformer with encoder-decoder structure
- **Attention Mechanism**: Multi-head self-attention and cross-attention
- **Training**: Adam optimizer with learning rate scheduling
- **Evaluation**: BLEU and METEOR metrics for translation quality
- **Data Format**: Supports standard parallel corpus formats

## Results

The model achieves strong performance on German-to-English translation tasks, with BLEU scores competitive with baseline Transformer implementations.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
