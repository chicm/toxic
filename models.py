
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import settings

class ToxicModel(nn.Module):
    def __init__(self):
        super(ToxicModel, self).__init__()
        self.name = 'ToxicModel'
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 1)
    
    def forward(self, x):
        _, pooled_out = self.bert_model(x)
        
        return self.fc(pooled_out)


def create_model(args):
    model = ToxicModel()
    model_file = os.path.join(settings.MODEL_DIR, model.name, args.ckp_name)

    parent_dir = os.path.dirname(model_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    print('model file: {}, exist: {}'.format(model_file, os.path.exists(model_file)))

    if args.predict and (not os.path.exists(model_file)):
        raise AttributeError('model file does not exist: {}'.format(model_file))

    if os.path.exists(model_file):
        print('loading {}...'.format(model_file))
        model.load_state_dict(torch.load(model_file))

    if torch.cuda.device_count() > 1:
        model_name = model.name
        model = DataParallel(model)
        model.name = model_name

    model = model.cuda()

    return model, model_file



if __name__ == '__main__':
    x = torch.tensor([[1,2,3,4,5]]).cuda()
    model = ToxicModel().cuda()

    y = model(x)
    print(y)
