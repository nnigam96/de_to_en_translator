from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from model_package.data_utils import *
from model_package.constants import *

# Modify the URLs for the dataset
multi30k.URL["train"] = TRAIN_URL 
multi30k.URL["valid"] = VAL_URL 

def get_loaders(token_transform, vocab_transform, BATCH_SIZE=BATCH_SIZE):
    """
    Get data loaders for training and validation datasets.

    Args:
        token_transform (dict): Token transforms for different languages.
        vocab_transform (dict): Vocab transforms for different languages.
        BATCH_SIZE (int, optional): Batch size. Defaults to BATCH_SIZE.

    Returns:
        DataLoader: Training data loader.
        DataLoader: Validation data loader.
    """
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln],  # Tokenization
                                                   vocab_transform[ln],  # Numericalization
                                                   tensor_transform)      # Add BOS/EOS and create tensor

    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=lambda batch: collate_fn(batch, text_transform=text_transform))

    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=lambda batch: collate_fn(batch, text_transform=text_transform))

    return train_dataloader, val_dataloader
