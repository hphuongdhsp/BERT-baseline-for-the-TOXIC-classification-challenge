#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:23:10 2019

@author: ai
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch

import torch.utils.data
from utils_bert import convert_lines
from tqdm import tqdm

from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
from pytorch_pretrained_bert import BertConfig
import argparse

parser = argparse.ArgumentParser(description='Salt Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 10)')
parser.add_argument('--max_sequence_length', type=int, default=128, metavar='LS',
                        help='length of traing sequence (default: 10)')
parser.add_argument('--lr', type=float, default=2e-5, metavar='LR',
                        help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--model_name', type=str, default='uncased_L-12_H-768_A-12', metavar='DD',
                    help='Where the dataset is saved to.')
parser.add_argument('--data_dir', type=str, default='/media/ai/376A1D136649692B/toxic/input', metavar='DM',
                    help='Where the dataset is saved to.')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
parser.add_argument("--pytorch_dump_path",
                        default = '/media/ai/376A1D136649692B/pretrained_bert/pretrain_bert_pytorch/',
                        type = str,metavar='DT',help = "Path to the output PyTorch model.")
parser.add_argument("--tf_checkpoint_path",
                        default = '/media/ai/376A1D136649692B/pretrained_bert/tensorflow_pretrained/',
                        type = str,metavar='DP',help = "Path the TensorFlow checkpoint path.")
parser.add_argument("--weight_path_pytorch",
                        default = '/media/ai/376A1D136649692B/toxic/input/weight/',
                        type = str,metavar='DP',help = "Path the Pytorch weight")

args = parser.parse_args()
device=torch.device('cuda')

WORK_DIR = os.path.join(args.pytorch_dump_path,args.model_name)
TOXICITY_COLUMN = 'target'

bert_config = BertConfig(os.path.join(args.tf_checkpoint_path,args.model_name,'bert_config.json'))

### tokenizer
BERT_MODEL_PATH=os.path.join(args.tf_checkpoint_path,args.model_name)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)


test_df = pd.read_csv(os.path.join(args.data_dir,"test.csv"))
test_df['comment_text'] = test_df['comment_text'].astype(str) 
X_test = convert_lines(test_df["comment_text"].fillna("DUMMY_VALUE"), args.max_sequence_length,tokenizer)

model = BertForSequenceClassification(bert_config, num_labels=1)
model.load_state_dict(torch.load(os.path.join(args.weight_path_pytorch,args.model_name+"bert_pytorch.bin")))
model.to(device)
for param in model.parameters():
    param.requires_grad = False
model.eval()

test_preds = np.zeros((len(X_test)))
test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))
test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)
tk0 = tqdm(test_loader)
for i, (x_batch,) in enumerate(tk0):
    pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
    test_preds[i * 32:(i + 1) * 32] = pred[:, 0].detach().cpu().squeeze().numpy()

test_pred = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()



