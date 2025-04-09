# German-to-English Translator

A powerful and efficient German-to-English translation system built using Transformer models. This project provides both a training pipeline for custom model development and a ready-to-use inference API for translation tasks.

![Translation Demo](https://github.com/nnigam96/modlee_code_test/assets/99565294/4b1fb625-fefb-405d-bb4e-29e457a1a21f)

## Features

- **Transformer-based Architecture**: Utilizes state-of-the-art transformer models for accurate translations
- **Easy-to-Use API**: Simple REST API for quick integration into applications
- **Custom Training Pipeline**: Full training pipeline for model customization
- **Modular Design**: Separated into model training and inference packages
- **SpaCy Integration**: Uses SpaCy for efficient text tokenization
- **Test Feature**: Testing Hugging Face LLM summarization workflow
- **Test Feature**: Testing summarization with German-to-English translator project

## Project Structure

```
de_to_en_translator/
├── model_package/          # Model training and architecture
│   ├── model_package/      # Core model implementation
│   │   ├── model.py        # Transformer model definition
│   │   ├── data.py         # Data loading and preprocessing
│   │   ├── trainer.py      # Training routines
│   │   └── inference.py    # Inference implementation
│   └── setup.py            # Package installation
├── infer_package/          # Inference package
│   ├── infer_package/      # Translation implementation
│   │   └── translate.py    # Translation interface
│   └── setup.py            # Package installation
└── demo_api/               # Demo API
    └── app.py              # Flask API implementation
```

## Quick Start

### Prerequisites

```bash
# Install language models for tokenization
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

### Installation

1. Install the model package:
```bash
cd model_package
pip install dist/model_package-0.1.tar.gz
```

2. Install the inference package:
```bash
cd infer_package
pip install dist/infer_package-0.1.tar.gz
```

### Usage

#### Translation API

```python
import infer_package.translate as translator

# Translate German text to English
result = translator.get_translation("Ich bin ein Berliner")
print(result)  # Output: 'I am a Berliner'
```

#### Running the Demo API

```bash
cd demo_api/demo_api
python app.py
```

The API will be available at `http://localhost:5000`

## Development

### Training Custom Models

1. Navigate to the model package:
```bash
cd model_package/model_package
```

2. Configure model parameters in `constants.py`

3. Start training:
```bash
python driver.py
```

Trained models are saved in `model_package/model_package/checkpoints/`

### Package Development

#### Creating Model Package
```bash
cd model_package
python setup.py sdist bdist_wheel
```

#### Creating Inference Package
```bash
cd infer_package
python setup.py sdist bdist_wheel
```

## Technical Details

- **Model Architecture**: Transformer-based neural network
- **Tokenization**: SpaCy for German and English text processing
- **API Framework**: Flask for REST API implementation
- **Training Pipeline**: Custom training loop with configurable parameters

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
