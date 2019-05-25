# BERT-baseline-for-the-TOXIC-classification-challenge
This repository is the BERT benchmark for the classification.

**BERT** was released together with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

Here we use data of the Jigsaw Unintended Bias in Toxicity Classification challenge whose mission is to detect toxicity across a diverse range of conversations(https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification).




## Dependencies
- tqdm
- numpy
- pickle
- scikit-learn
- PyTorch 1.0
- apx nvidia
- pytorch_pretrained_bert
- matplotlib
- pandas

## How to use the code

- We first down load pretrained bert model( for instance: //www.kaggle.com/maxjeblick/bert-pretrained-models)

- Download data from https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data. The data of this project  will be organized as: 
├── input
|  └── train.cvs
|  └── test.cvs 


- Run convert_tf_checkpoint_to_pytorch.py to convert the tensofflow weight into the Pytorch weight. 

- Run bert_traing.py to start to trainning

- To interface, we use bert_interface.py

