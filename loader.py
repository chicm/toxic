import os, glob
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
from sklearn.utils import shuffle
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from preprocess import preprocess_text

import settings

MAX_LEN = 200
aux_columns = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
'''
def preprocess(data):
    
    #Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    #CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    return data
'''

class ToxicDataset(data.Dataset):
    def __init__(self, df,  train_mode=True, labeled=True):
        super(ToxicDataset, self).__init__()
        self.df = df
        self.train_mode = train_mode
        self.labeled = labeled
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_token_ids(self, text):
        tokens = ['[CLS]'] + self.tokenizer.tokenize(str(text))[:MAX_LEN-2] + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) < MAX_LEN:
            token_ids += [0] * (MAX_LEN - len(token_ids))
        return torch.tensor(token_ids[:MAX_LEN])

    def get_label(self, row):
        return int(row.target >= 0.5), torch.tensor((row[aux_columns].values >= 0.5).astype(np.int16)), row.weights

    def __getitem__(self, index):
        row = self.df.iloc[index]
        token_ids = self.get_token_ids(row.comment_text) 
        if self.labeled:
            labels = self.get_label(row)
            return token_ids, labels[0], labels[1], labels[2]
        else:
            return token_ids

    def __len__(self):
        return len(self.df)

    def collate_fn(self, batch):
        if self.labeled:
            token_ids = torch.stack([x[0] for x in batch])
            labels = torch.tensor([x[1] for x in batch])
            aux_labels = torch.stack([x[2] for x in batch])
            weights = torch.tensor([x[3] for x in batch])
            return token_ids, labels, aux_labels, weights
        else:
            return torch.stack(batch)

def add_loss_weight(df):
    # Overall
    weights = np.ones((len(df),)) / 4
    # Subgroup
    weights += (df[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4
    # Background Positive, Subgroup Negative
    weights += (( (df['target'].values>=0.5).astype(bool).astype(np.int) +
       (df[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
    # Background Negative, Subgroup Positive
    weights += (( (df['target'].values<0.5).astype(bool).astype(np.int) +
       (df[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
    #loss_weight = 1.0 / weights.mean()
    df['weights'] = weights

def get_train_val_loaders(batch_size=64, val_batch_size=256, val_percent=0.95, val_num=10000):
    #df = shuffle(pd.read_csv(os.path.join(settings.DATA_DIR, 'train_clean.csv')), random_state=1234)
    df = shuffle(pd.read_csv(os.path.join(settings.DATA_DIR, 'train.csv')), random_state=1234)
    #print(df.head())
    df.comment_text = preprocess_text(df.comment_text)
    add_loss_weight(df)
    
    print(df.shape)

    split_index = int(len(df) * val_percent)

    df_train = df[:split_index]
    df_val = df[split_index:]

    if val_num is not None:
        df_val = df_val[:val_num]
    
    print(df_train.head())
    print(df_val.head())

    ds_train = ToxicDataset(df_train)
    train_loader = data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=ds_train.collate_fn, drop_last=True)
    train_loader.num = len(df_train)

    ds_val = ToxicDataset(df_val)
    val_loader = data.DataLoader(ds_val, batch_size=val_batch_size, shuffle=False, num_workers=4, collate_fn=ds_val.collate_fn, drop_last=False)
    val_loader.num = len(df_val)
    val_loader.df = df_val

    return train_loader, val_loader

def get_test_loader(batch_size):
    #df = pd.read_csv(os.path.join(settings.DATA_DIR, 'test_clean.csv'))
    df = pd.read_csv(os.path.join(settings.DATA_DIR, 'test.csv'))
    #print(df.head())
    df.comment_text = preprocess_text(df.comment_text)
    #print(df.head())
    ds_test = ToxicDataset(df, train_mode=False, labeled=False)
    loader = data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=ds_test.collate_fn, drop_last=False)
    loader.num = len(df)

    return loader

def test_train_loader():
    loader, _ = get_train_val_loaders(4)
    for ids, labels, aux_labels, weights in loader:
        print(ids)
        print(labels)
        print(aux_labels)
        print(weights)
        break

def test_test_loader():
    loader = get_test_loader(4)
    for ids in loader:
        print(ids.shape)
        print(ids)
        #print(labels)
        break

if __name__ == '__main__':
    test_train_loader()
    #test_test_loader()
