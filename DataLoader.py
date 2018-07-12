
# coding: utf-8

import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
from tqdm import tqdm
import pickle
import torch.nn as nn

class DataLoader:
    def __init__(self, batch_size=30, device=-1):
        self.batch_size=batch_size
        self.device = device
        #Define fields
        TEXT = data.Field(lower=True, include_lengths=False, batch_first=True)
        CHAR = data.Field(lower=True, include_lengths=False, batch_first=True, tokenize=list)
        TEXT_C = data.NestedField(CHAR)
        LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)
        INDEX = data.Field(sequential=False, use_vocab=False, batch_first=True)
        ID = data.RawField()
        fields = [("context",TEXT),("query",TEXT),("label",LABEL),("context_c",TEXT_C),("query_c",TEXT_C),("index",INDEX)]
        train_data = []
        val_data = []
        dev_data = []

        #Generate examples
        print("Loading datasets...")
        print("Loading training set...")
        try:
            with open("./data/processed/train_set.data", 'rb') as f:
                train_data = pickle.load(f)
            print("Loaded training set from file.")
        except:
            print("Failed to loaded training set from file. Processing training data...")
            with open("./data/squad/train.context") as f:
                context = list(f)

            with open("./data/squad/train.question") as f:
                query = list(f)

            with open("./data/squad/train.span") as f:
                label = list(f)
                for i in range(len(label)):
                    splited = list(map(int,label[i].split()))
                    label[i]=splited

            for i in tqdm(range(len(context)),ascii=True):
                list_content = [context[i],query[i],label[i],context[i],query[i],i]
                train_ex = data.Example.fromlist(list_content,fields)
                train_data.append(train_ex)
            with open("./data/processed/train_set.data",'wb') as f:
                pickle.dump(train_data,f)
        train_set = data.Dataset(train_data,fields)

        print("Loading dev set...")
        try:
            with open("./data/processed/dev_set.data", 'rb') as f:
                dev_data = pickle.load(f)
            print("Loaded dev set from file.")
        except:
            print("Failed to loaded dev set from file. Processing dev data...")
            with open("./data/squad/dev.context") as f:
                context = list(f)

            with open("./data/squad/dev.question") as f:
                query = list(f)

            with open("./data/squad/dev.span") as f:
                label = list(f)
                for i in range(len(label)):
                    splited = list(map(int,label[i].split()))
                    label[i]=splited

            for i in tqdm(range(len(context)),ascii=True):
                list_content = [context[i],query[i],label[i],context[i],query[i],i]
                dev_ex = data.Example.fromlist(list_content,fields)
                dev_data.append(dev_ex)
            with open("./data/processed/dev_set.data",'wb') as f:
                pickle.dump(dev_data,f)
        dev_set = data.Dataset(dev_data,fields)

        print("Loading validation set...")
        try:
            with open("./data/processed/val_set.data", 'rb') as f:
                val_data = pickle.load(f)
            print("Loaded validation set from file.")
        except:
            print("Failed to loaded validation set from file. Processing validation data...")
            with open("./data/squad/val.context") as f:
                context = list(f)

            with open("./data/squad/val.question") as f:
                query = list(f)

            with open("./data/squad/val.span") as f:
                label = list(f)
                for i in range(len(label)):
                    splited = list(map(int,label[i].split()))
                    label[i]=splited

            for i in tqdm(range(len(context)),ascii=True):
                list_content = [context[i],query[i],label[i],context[i],query[i],i]
                val_ex = data.Example.fromlist(list_content,fields)
                val_data.append(val_ex)
            with open("./data/processed/val_set.data",'wb') as f:
                pickle.dump(val_data,f)
        val_set = data.Dataset(val_data,fields)


        print("Loading word embeddings...")
        glove_vecs = GloVe(name='6B',dim=100)
        glove_vecs.unk_init = nn.init.xavier_uniform_

        print("Building vocabulary...")
        TEXT.build_vocab(train_set, vectors=glove_vecs)
        TEXT_C.build_vocab(train_set,min_freq=20)
        self.vocab_vec = TEXT.vocab.vectors
        print(len(self.vocab_vec)," words in word vocabulary.")
        self.char_size = len(TEXT_C.vocab)
        print(len(TEXT_C.vocab)," tokens in char vocabulary.")

        print("Generating iterator...")
        self.train_iter= iter(data.Iterator(train_set, batch_size=self.batch_size,device=self.device,sort_key=lambda x:len(x.context),repeat=True,sort=True))
        self.dev_iter = iter(data.Iterator(dev_set, batch_size=self.batch_size,device=self.device,sort_key=lambda x:len(x.context),repeat=False,sort=True))
        self.val_iter = iter(data.Iterator(val_set, batch_size=self.batch_size,device=self.device,sort_key=lambda x:len(x.context),repeat=True,sort=True))
        print("DataLoader initiated.")

    def next_train(self):
        try:
            train_data = next(self.train_iter)
        except:
            raise StopIteration
        else:
            return train_data

    def dev(self):
        return self.dev_iter

    def next_val(self):
        try:
            val_data = next(self.val_iter)
        except:
            raise StopIteration
        else:
            return val_data


# In[65]:
