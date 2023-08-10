from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.utils.data import DataLoader
import torch 
from model_package.utils import generate_square_subsequent_mask, create_mask
from model_package.constants import *
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Tokenization and Vocabulary Placeholder Initialization
token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

# Method to train a single epoch of the Seq2Seq model
def train_epoch(model, optimizer, loss_fn, train_loader):
    """
    Train the Seq2Seq model for a single epoch.

    Args:
        model (nn.Module): The Seq2Seq model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        loss_fn: The loss function for calculating training loss.
        train_loader (DataLoader): DataLoader for the training dataset.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    losses = 0
    
    for src, tgt in train_loader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_loader))

# Method to evaluate the Seq2Seq model
def evaluate(model, loss_fn, val_loader):
    """
    Evaluate the Seq2Seq model.

    Args:
        model (nn.Module): The Seq2Seq model to be evaluated.
        loss_fn: The loss function for calculating evaluation loss.
        val_loader (DataLoader): DataLoader for the validation dataset.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    losses = 0

    for src, tgt in val_loader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_loader))

# Method to train the Seq2Seq model for multiple epochs and save checkpoints
def model_train(model, optimizer, loss_fn, epochs, train_loader, val_loader):
    """
    Train the Seq2Seq model for multiple epochs.

    Args:
        model (nn.Module): The Seq2Seq model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        loss_fn: The loss function for calculating training and evaluation loss.
        epochs (int): Number of training epochs.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')  # Initialize with a high value

    for epoch in range(1, epochs+1):
        start_time = timer()
        train_loss = train_epoch(model, optimizer, loss_fn, train_loader)
        end_time = timer()
        val_loss = evaluate(model, loss_fn, val_loader)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save model checkpoint if validation loss decreases
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join('D:\Git Repos\modlee_code_test\checkpoints', 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
    
    # Plot and save the training and validation loss curve
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs+1), val_losses)
