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
from googletrans import Translator
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AdamW, T5ForConditionalGeneration, \
    T5Tokenizer, get_linear_schedule_with_warmup

from elt import translit
import pandas as pd
import pickle
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, \
    SequentialSampler
import os
from functions import *
from transformers import T5Tokenizer, T5ForConditionalGeneration
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


# import faiss

def most(l):
    a = max(set(l), key=l.count)
    if a == 'entailment':
        return 'true'
    if a == 'contradiction':
        return 'false'
    if a == 'neutral':
        return a


def concat(l1, l2):
    return l1 + ' : ' + l2


def rename(a):
    if a == 'entailment':
        return 'true'
    if a == 'contradictory':
        return 'false'
    if a == 'neutral':
        return a


SEED = 41
set_seed(SEED)

# define a rich console logger

console = Console(record=True)

train_df = pd.DataFrame()
test_df = pd.DataFrame()
val_df = pd.DataFrame()

# building dataframe

for file in os.listdir('./snli_1.0'):
    if file == 'snli_1.0_train_format.jsonl':
        f = open('./snli_1.0/' + file)
        data = json.load(f)
        f.close()

        train_df['input'] = [concat(i['sentence1'], i['sentence2'])
                             for i in data][:10000]
        train_df['output'] = [most(i['annotator_labels']) for i in
                              data][:10000]

    if file == 'snli_1.0_dev_format.jsonl':
        f = open('./snli_1.0/' + file)
        data = json.load(f)
        f.close()

        val_df['input'] = [concat(i['sentence1'], i['sentence2'])
                           for i in data][:1000]
        val_df['output'] = [most(i['annotator_labels']) for i in
                            data][:1000]
    if file == 'snli_1.0_test_format.jsonl' and sys.argv[1] == 'demo':
        with open('demo.txt', 'r') as f:
            inputs = f.readlines()

            inp = []
            out = []

            for i in range(int(len(inputs)/2)):
                in1 = inputs[2*i].strip()
                in2 = inputs[2*i+1].strip()

                if sys.argv[2] == 'hindi':
                    to_hindi = translit('hindi')

                    o1 = to_hindi.convert([in1])
                    o2 = to_hindi.convert([in2])
                    translator = Translator()
                    i1 = translator.translate(o1[0], src='hi', dest='en')
                    i2 = translator.translate(o2[0], src='hi', dest='en')

                    inp.extend([concat(i1.text, i2.text)])

                else:
                    inp.extend([concat(in1,in2)])
                out.extend(['neutral'])

            test_df['input'] = inp
            test_df['output'] = out

    elif file == 'snli_1.0_test_format.jsonl' and sys.argv[1] == 'test_hindi':
        data = pickle.load(open('hindi_data2.pkl', 'rb'))
        test_df['input'] = [concat(i[1], i[2]) for i in data][:1000]
        test_df['output'] = [rename(i[0]) for i in data][:1000]
    elif file == 'snli_1.0_test_format.jsonl':
        f = open('./snli_1.0/' + file)
        data = json.load(f)
        f.close()

        test_df['input'] = [concat(i['sentence1'], i['sentence2'])
                            for i in data][:2000]
        test_df['output'] = [most(i['annotator_labels']) for i in
                             data][:2000]

val_df = val_df.reset_index(drop=True)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# print train_df
# print val_df
# print(test_df)

# exit()

model_name = 't5-base'

logger = logging.getLogger(__name__)

args_dict = dict(  # path for data files
                   # path to save the checkpoints
                   # if you want to enable 16-bit training then install apex and set this to true
                   # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
                   # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    data_dir='',
    output_dir='output',
    model_name_or_path=model_name,
    tokenizer_name_or_path=model_name,
    max_seq_length=128,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=32,
    eval_batch_size=32,
    num_train_epochs=10,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False,
    opt_level='O1',
    max_grad_norm=1.0,
    seed=SEED,
    )

tokenizer = T5Tokenizer.from_pretrained(model_name)

dataset = DatasetProcessor(tokenizer, val_df, max_len=128)

args = argparse.Namespace(**args_dict)

checkpoint_callback = \
    pl.callbacks.ModelCheckpoint(filepath=args.output_dir,
                                 monitor='val_loss', mode='min',
                                 save_top_k=1)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision=(16 if args.fp_16 else 32),
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
    )

model = T5FineTuner(args, train_df, val_df)
trainer = pl.Trainer(**train_params)
from transformers import T5ForConditionalGeneration, T5Tokenizer

if sys.argv[1] == 'train':
    trainer.fit(model)
elif sys.argv[1] in ['test', 'test_hindi']:
    ckpt = torch.load('output/epoch=2.ckpt')
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(torch.device(('cuda'
                      if torch.cuda.is_available else 'cpu')))
elif sys.argv[1] == 'demo':
    ckpt = torch.load('output/epoch=2.ckpt')
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(torch.device(('cuda'
                      if torch.cuda.is_available else 'cpu')))
import textwrap
from tqdm.auto import tqdm
from sklearn import metrics

dataset = DatasetProcessor(tokenizer, test_df, max_len=128)

loader = DataLoader(dataset, batch_size=32, num_workers=4,
                    shuffle=False)
model.model.eval()
outputs = []
targets = []
for batch in tqdm(loader):
    outs = model.model.generate(input_ids=batch['source_ids'].cuda(),
                                attention_mask=batch['source_mask'
                                ].cuda(), max_length=2, output_scores = True, return_dict_in_generate = True)
    ids = list(list(tokenizer("true false neutral", return_tensors = 'pt').input_ids)[0].cpu().detach().numpy())[:-1]
    # print(ids)
    # print(outs)
    # print(outs.scores[0].shape)
    scores = torch.transpose(torch.vstack([outs.scores[0][:, id] for id in ids]), 0, 1)
    print(scores)
    print(torch.nn.Softmax()(torch.tensor(scores)))
    exit()
    dec = [tokenizer.decode(ids, skip_special_tokens=True,
           clean_up_tokenization_spaces=True) for ids in outs]
    target = [tokenizer.decode(ids, skip_special_tokens=True,
              clean_up_tokenization_spaces=True) for ids in
              batch['target_ids']]
    outputs.extend(dec)
    targets.extend(target)

    print(dec)
    for i in dec:
        if i == 'false':
            print('output: ' + str('contradictory'))
        elif i == 'true':
            print('output: ' + str('entailment'))
        else:
            print('output: ' + str('neutral'))
metrics.accuracy_score(targets, outputs)

if sys.argv[1] != 'demo':
    print(metrics.classification_report(targets, outputs))
