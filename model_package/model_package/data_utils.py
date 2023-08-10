from torch.nn.utils.rnn import pad_sequence
from model_package.constants import *
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k

# Place-holders
token_transform = {}
vocab_transform = {}

# Helper function to sequentially apply a list of transforms
def sequential_transforms(*transforms):
    """
    Sequentially apply a list of transforms to input data.

    Args:
        *transforms: List of transforms to apply.

    Returns:
        callable: Function that applies the transforms to input data.
    """
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# Function to add BOS/EOS tokens and create a tensor for input sequence indices
def tensor_transform(token_ids: list[int]):
    """
    Add BOS/EOS tokens and create a tensor for input sequence indices.

    Args:
        token_ids (list[int]): List of token indices.

    Returns:
        torch.Tensor: Tensor containing BOS, token indices, and EOS.
    """
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# Get ``src`` and ``tgt`` language text transforms to convert raw strings into tensor indices
def get_text_transform():
    """
    Get text transforms for source and target languages.

    Returns:
        dict: Dictionary of text transforms for different languages.
    """
    token_transform, vocab_transform = get_vocab_transforms()
    text_transform = {}
    
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln],  # Tokenization
                                                   vocab_transform[ln],  # Numericalization
                                                   tensor_transform)      # Add BOS/EOS and create tensor
    return text_transform

# Function to collate data samples into batch tensors
def collate_fn(batch, text_transform):
    """
    Collate data samples into batch tensors.

    Args:
        batch (list): List of data samples.
        text_transform (dict): Dictionary of text transforms.

    Returns:
        tuple: Tuple of source batch tensor and target batch tensor.
    """
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

# Get token and vocab transforms
def get_vocab_transforms(token_transform={}, vocab_transform={}):
    """
    Get token and vocab transforms for different languages.

    Args:
        token_transform (dict, optional): Existing token transforms. Defaults to empty dict.
        vocab_transform (dict, optional): Existing vocab transforms. Defaults to empty dict.

    Returns:
        tuple: Tuple of token and vocab transforms.
    """
    token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
    token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

    # Helper function to yield list of tokens
    def yield_tokens(data_iter, language: str):
        language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

        for data_sample in data_iter:
            yield token_transform[language](data_sample[language_index[language]])

    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']  # Special symbols for vocab
    
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)
        vocab_transform[ln].set_default_index(UNK_IDX)

    return token_transform, vocab_transform
