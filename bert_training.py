#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 09:21:52 2019

@author: ai
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch

import torch.utils.data
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from utils_bert import convert_lines,seed_everything,createFolder
from tqdm import tqdm, tqdm_notebook
from apex import amp
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification,BertAdam
from pytorch_pretrained_bert import BertConfig
import argparse

parser = argparse.ArgumentParser(description='Salt Example')
parser.add_argument('--batch-size', type=int, default=5, metavar='N',
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
                        default = '/media/ai/376A1D136649692B/toxic/input/weight',
                        type = str,metavar='DP',help = "Path the Pytorch weight")


args = parser.parse_args()

device=torch.device('cuda')

WORK_DIR = os.path.join(args.pytorch_dump_path,args.model_name)
                          #Validation Size
TOXICITY_COLUMN = 'target'

bert_config = BertConfig(os.path.join(args.tf_checkpoint_path,args.model_name,'bert_config.json'))

### tokenizer
BERT_MODEL_PATH=os.path.join(args.tf_checkpoint_path,args.model_name)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)
##########READ DATA ################

nrow =1000
train_df = pd.read_csv(os.path.join(args.data_dir,"train.csv"),nrows= nrow)
train_df['comment_text'] = train_df['comment_text'].astype(str) 
sequences_train = convert_lines(train_df["comment_text"].fillna("DUMMY_VALUE"),args.max_sequence_length,tokenizer)
train_df=train_df.fillna(0)

y_columns=['target']

train_df = train_df.drop(['comment_text'],axis=1)
# convert target to 0,1
train_df['target']=(train_df['target']>=0.5).astype(float)
y = train_df[y_columns].values

##############################DATALOADER#######################################
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(sequences_train, y,
                                                    stratify=y, 
                                                    test_size=0.25)
###############################################################################



seed_everything(seed=args.seed)
accumulation_steps=1
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train,dtype=torch.long), torch.tensor(y_train,dtype=torch.float))
valid_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val,dtype=torch.long),torch.tensor(y_val,dtype=torch.float))

###################################################33




###########MODEL #########################
model = BertForSequenceClassification.from_pretrained(os.path.join(args.pytorch_dump_path,args.model_name),
                                                      cache_dir=None,num_labels=len(y_columns))

model.zero_grad()
model = model.to(device)
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


num_train_optimization_steps = int(args.epochs*len(train_dataset)/args.batch_size/accumulation_steps)

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=args.lr,
                     warmup=0.05,
                     t_total=num_train_optimization_steps)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
model=model.train()

output_model_file= os.path.join(args.data_dir,'weight',args.model_name)
createFolder(args.data_dir+'/weight')
 ###############################################################################
tq = tqdm_notebook(range(args.epochs))
for epoch in tqdm_notebook(range(args.epochs)):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    avg_loss = 0.
    avg_accuracy = 0.
    lossf=None
    tk0 = tqdm_notebook(enumerate(train_loader),total=len(train_loader),leave=False)
    for i,(x_batch, y_batch) in tk0:
        optimizer.zero_grad()
        y_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
        loss =  F.binary_cross_entropy_with_logits(y_pred,y_batch.to(device))
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
            optimizer.step()                            # Now we can do an optimizer step
            optimizer.zero_grad()
        if lossf:
            lossf = 0.98*lossf+0.02*loss.item()
        else:
            lossf = loss.item()
        tk0.set_postfix(loss = lossf)
        avg_loss += loss.item() / len(train_loader)
        avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float) ).item()/len(train_loader)
    tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)
    torch.save(model.state_dict(), output_model_file+"bert_pytorch.bin")
    model.eval()
    valid_preds = np.zeros((len(X_val)))
    for i,(x_batch,y_batch)  in enumerate(tqdm(valid_loader)):
        val_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
        valid_preds[i*args.batch_size:(i+1)*args.batch_size]=val_pred[:,0].detach().cpu().squeeze().numpy()
    scores=roc_auc_score(y_val, valid_preds)
    print("ROC_AUC_SCORE is: ",scores)

