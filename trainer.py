from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.utils.data import DataLoader
import torch 
from utils import generate_square_subsequent_mask, create_mask
from constants import *
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'


# Place-holders
token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

def train_epoch(model, optimizer, loss_fn, train_loader):
    model.train()
    losses = 0
    #train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    #train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

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


def evaluate(model, loss_fn, val_loader):
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


#for epoch in range(1, NUM_EPOCHS+1):
#    start_time = timer()
#    train_loss = train_epoch(transformer, optimizer)
#    end_time = timer()
#    val_loss = evaluate(transformer)
#    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))


#train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
#val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

def model_train(model, optimizer, loss_fn, epochs, train_loader, val_loader):
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
    
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    loss_curve_path = os.path.join('D:\Git Repos\modlee_code_test\checkpoints', 'loss_curve.png')
    plt.savefig(loss_curve_path)
    plt.show()