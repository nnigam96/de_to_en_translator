from constants import *
from model import *
from data import *
from data_utils import *
from trainer import *
from torch.utils.data import DataLoader
from utils import *
from inference import *
import argparse
import torch

#token_transform, vocab_transform = get_vocab_transforms()

#SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
#TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

#model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
#                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
#model.load_state_dict(torch.load(r"D:\Git Repos\modlee_code_test\checkpoints\best_model.pth"))
#model = model.to(DEVICE)

#ans = translate(model, "Eine Gruppe von Menschen steht vor einem Iglu.")

one = "This sentence is in english"
two = "This sentence is in english"
score = metrics(one, two, type = "METEOR")
print(metrics(one, two))

#print(ans)