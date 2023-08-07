from constants import *
from model import *
from data import *
from data_utils import *
from trainer import *
from torch.utils.data import DataLoader
from utils import *

import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description='Driver for training and evaluating Transformer model on a English to German translation task')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs to train the model for')
    parser.add_argument('--num-encoder-layers', type=int, default=3, help='Number of encoder layers in the Transformer model')



    token_transform, vocab_transform = get_vocab_transforms()
    train_dataloader, val_dataloader = get_loaders(token_transform, vocab_transform)

    
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    transformer = transformer.to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    model_train(transformer, optimizer, loss_fn, 50, train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()