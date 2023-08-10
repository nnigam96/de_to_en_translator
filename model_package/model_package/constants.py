# Source and target languages
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Embedding size, number of heads, and feedforward hidden dimension for the transformer
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512

# Batch size for training and validation
BATCH_SIZE = 128

# Number of encoder and decoder layers in the transformer
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

# Token indices for special symbols
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

# List of special symbols in order of their indices
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']

# Number of epochs for training
NUM_EPOCHS = 18

# URLs for training and validation datasets
TRAIN_URL = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
VAL_URL = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

# Learning rate for optimizer
LEARNING_RATE = 0.0001

#METRIC_TYPES = ['BLEU', 'METEOR']