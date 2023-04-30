import argparse
import glob
import os
import json
import time
import logging
import random
import re
import sys
from itertools import chain
from string import punctuation
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
import pandas as pd
import pickle
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
from functions import *
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, MT5Model
from rich.table import Column, Table
from rich import box
from rich.console import Console
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from nltk.corpus import words
word_list = set(words.words())
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.data import find
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
import pickle
# import faiss

sent = ""

def most(l):
    a =  max(set(l), key = l.count)
    if a == "entailment":
        return "true"
    if a == "contradiction":
        return "false"
    if a == "neutral":
        return a

def rename(a):
    if a == "entailment":
        return "yes"
    if a == "contradictory":
        return "no"
    if a == "neutral":
        return a



def concat(l1, l2):
    global sent
    if sent == "":
        sent = l1
    return l1+" : "+l2


SEED = 41
set_seed(SEED)

# define a rich console logger
console=Console(record=True)

train_df = pd.DataFrame()
test_df = pd.DataFrame()
val_df = pd.DataFrame()

# building dataframe
with open('./parrot_big.pkl', 'rb') as f:
    data = pickle.load(f)
    data = [i for i in data if i[2] in ['contradictory', 'entailment', 'neutral']]
    np.random.shuffle(data)
    
    # print([concat(i[0],i[1]) for i in data][:2000])
    # print([rename(i[2]) for i in data][:2000])
    num_train = 15000
    num_test = 500
    num_val = 500

    train_df['input'] = [concat(i[0],i[1]) for i in data][:num_train]
    train_df['output'] = [rename(i[2]) for i in data][:num_train]

    val_df['input'] = [concat(i[0],i[1]) for i in data][num_train:num_train+num_val]
    val_df['output'] = [rename(i[2]) for i in data][num_train:num_train+num_val]

    positive_indices = []
    negative_indices = []
    neutral_indices = []
    for i in range(num_train + num_val, num_train + num_val + 10000):
        if data[i][2] == 'entailment':
            positive_indices.append(i)
        elif data[i][2] == 'contradictory':
            negative_indices.append(i)
        elif data[i][2] == 'neutral':
            neutral_indices.append(i)
        if len(positive_indices) > num_test and len(negative_indices) > num_test and len(neutral_indices) > num_test:
            break
    indices = positive_indices[:num_test] + negative_indices[:num_test] + neutral_indices[:num_test]
    np.random.shuffle(indices)
    test_df['input'] = [concat(data[i][0],data[i][1]) for i in indices]
    test_df['output'] = [rename(data[i][2]) for i in indices]



val_df = val_df.reset_index(drop=True)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


print(train_df)
print(val_df)
print(test_df)

# exit()

model_name = 'google/flan-t5-base'
model_name = 'bert-base-uncased'
model_name = 'google/mt5-small'

logger = logging.getLogger(__name__)

args_dict = dict(
    data_dir="", # path for data files
    output_dir='output', # path to save the checkpoints
    model_name_or_path=model_name,
    tokenizer_name_or_path=model_name,
    max_seq_length=128,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=10,
    eval_batch_size=10,
    num_train_epochs=10,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=SEED,
)


tokenizer =  T5Tokenizer.from_pretrained(model_name)

dataset = DatasetProcessor(tokenizer, val_df,  max_len=128)

args = argparse.Namespace(**args_dict)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=args.output_dir, monitor="val_loss", mode="min", save_top_k=1
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)

model = T5FineTuner(args, train_df, val_df)
ckpt = torch.load('output-backup/epoch=8.ckpt')
model.load_state_dict(ckpt['state_dict'])
model = model.to(torch.device(('cuda' if torch.cuda.is_available else 'cpu')))

# trainer = pl.Trainer(**train_params)
# trainer.fit(model)

import textwrap
from tqdm.auto import tqdm
from sklearn import metrics

dataset = DatasetProcessor(tokenizer, test_df,  max_len=128)

loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False)
model.model.eval()
outputs = []
targets = []
#for batch in tqdm(loader):
#    outs = np.argmax(model(input_ids=batch['source_ids'].cuda(), 
#                                attention_mask=batch['source_mask'].cuda()).cpu().detach().numpy(), axis = 1)
#    outputs.extend(outs)
#    targets.extend(list(batch['labels'].cpu().detach().numpy()))
    
#metrics.accuracy_score(targets, outputs)
#print(metrics.classification_report(targets, outputs))

device = torch.device(('cuda' if torch.cuda.is_available else 'cpu'))
inputs = ["तापमान बढ़ रहा है : आज गर्मी है"]
tokenized_inputs = tokenizer(inputs, return_tensors = 'pt').to(device)
outs = model(input_ids = tokenized_inputs.input_ids, attention_mask = tokenized_inputs.attention_mask)
print(outs)
