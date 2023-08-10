import torch
from model_package.constants import *
from model_package.data import *  # Import any necessary data-related modules here

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

# Function to generate a square subsequent mask for masking out future positions
def generate_square_subsequent_mask(sz):
    """
    Generate a square subsequent mask for masking out future positions in self-attention.

    Args:
        sz (int): Size of the mask.

    Returns:
        torch.Tensor: Square subsequent mask.
    """
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Function to create masks for source and target sequences
def create_mask(src, tgt):
    """
    Create masks for source and target sequences.

    Args:
        src (torch.Tensor): Source sequence.
        tgt (torch.Tensor): Target sequence.

    Returns:
        tuple: Tuple containing source mask, target mask, source padding mask, and target padding mask.
    """
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
