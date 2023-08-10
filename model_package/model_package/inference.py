from model_package.constants import *
import torch
from model_package.utils import *
from model_package.data_utils import *
from model_package.model import *
from sacrebleu import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import sentence_bleu

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    Generate an output sequence using the greedy decoding algorithm.

    Args:
        model (nn.Module): The Seq2Seq model for decoding.
        src (Tensor): Source sequence tensor.
        src_mask (Tensor): Source sequence mask.
        max_len (int): Maximum length of the output sequence.
        start_symbol (int): Index of the start symbol in the target vocabulary.

    Returns:
        Tensor: Generated target sequence.
    """
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# Function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    """
    Translate an input sentence into the target language.

    Args:
        model (nn.Module): The Seq2Seq model for translation.
        src_sentence (str): Input sentence to be translated.

    Returns:
        str: Translated sentence.
    """
    model.eval()
    text_transform  = get_text_transform()
    _, vocab_transform = get_vocab_transforms()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


# Function to compute evaluation metrics (BLEU or METEOR)
def metrics(reference, hypothesis, metric_type="BLEU"):
    """
    Compute evaluation metrics for translation quality.

    Args:
        reference (str): Reference sentence.
        hypothesis (str): Hypothesized sentence for evaluation.
        metric_type (str): Type of metric to compute (BLEU or METEOR).

    Returns:
        float: Computed metric score.
    """
    if metric_type == "BLEU":
        return sentence_bleu([reference], hypothesis)
    
    elif metric_type == "METEOR":
        return single_meteor_score([reference], [hypothesis])

def fetch_model(dir_path = r"D:\Git Repos\modlee_code_test\checkpoints\best_model.pth"):
    """
    Load model weights and hyperparameters for API.

    Args:
        dir_path (str): Path to saved model checkpoint.

    Returns:
        Dictionary with following keys:
        model: State dict of trained model.
        SRC_VOCAB_SIZE: Vocabulary size of source language.
        TGT_VOCAB_SIZE: Vocabulary size of target language.
    """
# Get token and vocab transforms for text processing
    _, vocab_transform = get_vocab_transforms()

    # Calculate vocabulary sizes for source and target languages
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

    # Create an instance of the Seq2SeqTransformer model
    model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                               NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    # Load the trained model's state dict
    if dir_path:
        model.load_state_dict(torch.load(dir_path, map_location=torch.device('cpu')))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # Move the model to the appropriate device (CPU or GPU)
    model = model.to(DEVICE)

    return {'model': model, 'src_vocab_size': SRC_VOCAB_SIZE, 'tgt_vocab_size': TGT_VOCAB_SIZE}