#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 13:41:05 2019

@author: ai
Because the BERT algorithm is released by Google and is trained by tensorplatform, 
then we need to change the pretrained weight from the tf platform to Pytorch platform. We so do

(1)We first download the pretrained and put it in tf_checkpoint_path (yon can download the weight form https://www.kaggle.com/maxjeblick/bert-pretrained-models)
(2) Converting the weight by this script, the weigth of the pytorch platform is stored in pytorch_dump_path
"""

import argparse
import shutil
import torch
from utils_bert import createFolder
import os
from pytorch_pretrained_bert.modeling import BertConfig, BertForPreTraining, load_tf_weights_in_bert

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--pytorch_dump_path",
                        default = '/media/ai/376A1D136649692B/pretrained_bert/pretrain_bert_pytorch/',
                        type = str,metavar='DT',help = "Path to the output PyTorch model.")
    parser.add_argument("--tf_checkpoint_path",
                        default = '/media/ai/376A1D136649692B/pretrained_bert/tensorflow_pretrained/',
                        type = str,metavar='DP',help = "Path the TensorFlow checkpoint path.")
    args = parser.parse_args()
    folders=os.listdir(args.tf_checkpoint_path)
    for i in folders:
        createFolder(os.path.join(args.pytorch_dump_path,i))
        
        convert_tf_checkpoint_to_pytorch(os.path.join(args.tf_checkpoint_path,i)+"/bert_model.ckpt",
                                     os.path.join(args.tf_checkpoint_path,i)+"/bert_config.json",
                                     os.path.join(args.pytorch_dump_path,i)+"/pytorch_model.bin")
        shutil.copyfile(os.path.join(args.tf_checkpoint_path,i) + "/bert_config.json", os.path.join(args.pytorch_dump_path,i)+ "/bert_config.json")
        