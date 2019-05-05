import os, glob
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
from sklearn.utils import shuffle
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import settings

MAX_LEN = 160

def preprocess(data):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    return data

class ToxicDataset(data.Dataset):
    def __init__(self, df,  train_mode=True, labeled=True):
        super(ToxicDataset, self).__init__()
        self.df = df
        self.train_mode = train_mode
        self.labeled = labeled
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_token_ids(self, text):
        tokens = self.tokenizer.tokenize('[CLS]' + text + '[SEP]')
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) < MAX_LEN:
            token_ids += [0] * (MAX_LEN - len(token_ids))
        return torch.tensor(token_ids[:MAX_LEN])

    def get_label(self, target):
        return int(target>0.5)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        token_ids = self.get_token_ids(row.comment_text) 
        if self.labeled:
            return token_ids, self.get_label(row.target)
        else:
            return token_ids

    def __len__(self):
        return len(self.df)

    def collate_fn(self, batch):
        if self.labeled:
            token_ids = torch.stack([x[0] for x in batch])
            labels = torch.tensor([x[1] for x in batch])
            return token_ids, labels
        else:
            return torch.stack(batch)

def get_train_val_loaders(batch_size=64, val_batch_size=256, val_percent=0.95, val_num=10000):
    df = shuffle(pd.read_csv(os.path.join(settings.DATA_DIR, 'train.csv')), random_state=1234)
    #print(df.head())
    df.comment_text = preprocess(df.comment_text)
    #print(df.head())
    print(df.shape)

    split_index = int(len(df) * val_percent)

    df_train = df[:split_index]
    df_val = df[:split_index:]

    if val_num is not None:
        df_val = df_val[:val_num]

    ds_train = ToxicDataset(df_train)
    train_loader = data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=ds_train.collate_fn, drop_last=True)
    train_loader.num = len(df_train)

    ds_val = ToxicDataset(df_val)
    val_loader = data.DataLoader(ds_val, batch_size=val_batch_size, shuffle=False, num_workers=4, collate_fn=ds_val.collate_fn, drop_last=False)
    val_loader.num = len(df_val)

    return train_loader, val_loader

def get_test_loader():
    pass

if __name__ == '__main__':
    loader, _ = get_train_val_loaders(4)
    for ids, labels in loader:
        print(ids.shape)
        print(labels)
        break