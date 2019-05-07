
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
        self.fc_aux = nn.Linear(768, 5)
    
    def forward(self, x):
        attention_mask = (x != 0)
        #print(attention_mask)
        layers, pooled_out = self.bert_model(x, attention_mask=attention_mask, output_all_encoded_layers=True)
        out = F.dropout(layers[-1][:, 0, :], p=0.2, training=self.training)
        
        return self.fc(out), self.fc_aux(out)

    def freeze(self):
        for param in self.bert_model.parameters():
            param.requires_grad = False
        #for param in self.fc.parameters():
        #    param.requires_grad = False

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

def convert_model():
    model = ToxicModel()
    model.load_state_dict(torch.load('/mnt/chicm/data/toxic/models/ToxicModel/best_model.pth_latest'), strict=False)
    torch.save(model.state_dict(), '/mnt/chicm/data/toxic/models/ToxicModel/best_model_new.pth')

def test_forward():
    x = torch.tensor([[1,2,3,4,5, 0, 0]]).cuda()
    model = ToxicModel().cuda()

    y = model(x)
    print(y)


if __name__ == '__main__':
    #convert_model()
    test_forward()