
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from pytorch_pretrained_bert import BertTokenizer, GPT2Tokenizer, BertForSequenceClassification, \
    BertAdam, GPT2Model, OpenAIAdam
from pytorch_pretrained_bert.modeling_gpt2 import GPT2PreTrainedModel

import settings

class GPT2ClassificationHeadModel(GPT2PreTrainedModel):
    def __init__(self, config, clf_dropout=0.4, n_class=8):
        super(GPT2ClassificationHeadModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.dropout = nn.Dropout(clf_dropout)
        self.linear = nn.Linear(config.n_embd * 2, n_class)

        nn.init.normal_(self.linear.weight, std = 0.02)
        nn.init.normal_(self.linear.bias, 0)
        
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        avg_pool = torch.mean(hidden_states, 1)
        max_pool, _ = torch.max(hidden_states, 1)
        h_conc = torch.cat((avg_pool, max_pool), 1)
        logits = self.linear(self.dropout(h_conc))
        return logits

sub_dir_dict = {
    'bert-base-uncased': 'base',
    'bert-base-cased': 'cased-base',
    'bert-large-uncased': 'large',
    'bert-large-cased': 'cased-large',
    'gpt2': 'gpt2'
}

def _create_model(args, num_classes=6):
    if args.use_path:
        weights_key = os.path.join(settings.BERT_WEIGHT_DIR, sub_dir_dict[args.model_name])
        assert os.path.isdir(weights_key)
    else:
        weights_key = args.model_name

    #if args.use_path:
    #    if 'large' in args.model_name:
    #        sub_dir = 'large'
    #    else:
    #        sub_dir = 'base'
    #    model = BertForSequenceClassification.from_pretrained(os.path.join(settings.BERT_WEIGHT_DIR, sub_dir),cache_dir=None,num_labels=num_classes)
    #    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    #else:
    if 'gpt2' in args.model_name:
        model = GPT2ClassificationHeadModel.from_pretrained(weights_key, clf_dropout=0.4, n_class=num_classes)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif 'bert' in args.model_name:
        model = BertForSequenceClassification.from_pretrained(weights_key, cache_dir=None, num_labels=num_classes)
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
    else:
        raise ValueError('model_name')

    return model, tokenizer
    

def create_model(args):
    model, tokenizer = _create_model(args, args.num_classes)

    model_file = os.path.join(settings.MODEL_DIR, '{}_{}_{}'.format(args.model_name, args.run_name, args.ifold), args.ckp_name)

    parent_dir = os.path.dirname(model_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    if args.init_ckp and os.path.exists(args.init_ckp):
        print('loading {}...'.format(args.init_ckp))
        model.load_state_dict(torch.load(args.init_ckp))

    print('model file: {}, exist: {}'.format(model_file, os.path.exists(model_file)))

    if args.predict and (not os.path.exists(model_file)):
        raise AttributeError('model file does not exist: {}'.format(model_file))

    if os.path.exists(model_file):
        print('loading {}...'.format(model_file))
        model.load_state_dict(torch.load(model_file))

    model = model.cuda()

    return model, model_file, tokenizer

def convert_model(args):
    args.num_classes = 6
    model, model_file, _ = create_model(args)
    if 'bert' in args.model_name:
        print(model.classifier.weight.size())
        print(model.classifier.bias.size())
        print(model.classifier.bias)
    elif 'gpt2' in args.model_name:
        print(model.linear.weight.size())
        print(model.linear.bias.size())
        print(model.linear.bias)
    torch.save(model.state_dict(), model_file+'_num6')

    args.ckp_name = 'tmp123'
    args.num_classes = 8
    new_model, _, _ = create_model(args)

    if 'bert' in args.model_name:
        new_model.classifier.weight[:6, :] = model.classifier.weight
        new_model.classifier.bias[:6] = model.classifier.bias
        new_model.bert = model.bert
    elif 'gpt2' in args.model_name:
        new_model.linear.weight[:6, :] = model.linear.weight
        new_model.linear.bias[:6] = model.linear.bias
        new_model.transformer = model.transformer
    torch.save(new_model.state_dict(), model_file)

def test_forward():
    import settings
    from apex import amp
    from pytorch_pretrained_bert import BertForSequenceClassification, BertAdam
    x = torch.tensor([[1,2,3,4,5, 0, 0]]*8).cuda()
    #model = ToxicModel().cuda()
    model = BertForSequenceClassification.from_pretrained(os.path.join(settings.BERT_WEIGHT_DIR, 'base'),cache_dir=None,num_labels=6).cuda()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters, lr=0.0001, warmup=0.1, t_total=100000)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
    model = DataParallel(model)
    model.train()
    print(x)
    y = model(x)
    print(y)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Landmark detection')
    parser.add_argument('--run_name', required=True, type=str, help='learning rate')
    parser.add_argument('--model_name', default='bert-base-uncased', type=str, help='learning rate')
    parser.add_argument('--ifold', default=0, type=int, help='lr scheduler patience')
    parser.add_argument('--use_path', action='store_true')
    parser.add_argument('--ckp_name', type=str, default='best_model.pth',help='check point file name')
    parser.add_argument('--init_ckp', default=None, type=str, help='resume from checkpoint path')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--convert', action='store_true')
    args = parser.parse_args()

    if args.convert:
        print('converting model from num_class 6 to 8')
        convert_model(args)
        print('done')
    #convert_model()
    #test_forward()
