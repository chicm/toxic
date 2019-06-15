import os
import argparse
import numpy as np
import pandas as pd
import logging as log
import time
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, _LRScheduler, ReduceLROnPlateau
import settings
from loader import get_train_val_loaders, get_test_loader
from models import create_model, convert_model
#from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert import BertTokenizer, GPT2Tokenizer, BertForSequenceClassification, \
    BertAdam, GPT2Model, OpenAIAdam
from pytorch_pretrained_bert.modeling_gpt2 import GPT2PreTrainedModel
from torch.nn import DataParallel
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from metrics import auc_score
from torch.nn import DataParallel
from apex import amp

MODEL_DIR = settings.MODEL_DIR


class FocalLoss(nn.Module):
    def forward(self, x, y):
        alpha = 0.25
        gamma = 2

        p = x.sigmoid()
        pt = p*y + (1-p)*(1-y)       # pt = p if t > 0 else 1-p
        w = alpha*y + (1-alpha)*(1-y)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        w = w.detach()
        #w.requires_grad = False
        #return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)
        return F.binary_cross_entropy_with_logits(x, y, w, reduction='none')

c = nn.BCEWithLogitsLoss(reduction='none')
f_c = FocalLoss()

def _reduce_loss(loss):
    #print('loss shape:', loss.shape)
    return loss.sum() / loss.shape[0]

def criterion(output, output_aux, target, target_aux, weights):
    loss1 = _reduce_loss(c(output, target.float()) * weights)
    loss2 = _reduce_loss(c(output_aux[:, :target_aux.size(1)], target_aux.float()) * weights.unsqueeze(-1))
    return loss1 * 5 + loss2

def train(args):
    print('start training...')
    #model, model_file = create_model(args)

    model, model_file, tokenizer = create_model(args)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    train_loader, val_loader = get_train_val_loaders(batch_size=args.batch_size, model_name=args.model_name, tokenizer=tokenizer, \
        ifold=args.ifold, clean_text=args.clean_text, val_batch_size=args.val_batch_size, val_num=args.val_num)

    num_train_optimization_steps = args.num_epochs * train_loader.num // train_loader.batch_size // 4

    #if args.optim_name == 'BertAdam':
    if 'bert' in args.model_name:
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.lr,
            warmup=args.warmup,
            t_total=num_train_optimization_steps
        )
    else:
        optimizer = OpenAIAdam(
            optimizer_grouped_parameters, 
            lr=args.lr,
            warmup=args.warmup,
            t_total=num_train_optimization_steps
        )

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    #model=model.train()

    best_f2 = 999.
    best_key = 'roc'

    print('epoch |    lr     |       %        |  loss  |  avg   |  loss  |  acc   |  prec  | recall |   roc  |  best  | time |  save |')

    if not args.no_first_val:
        val_metrics = validate(args, model, val_loader)
        print('val   |           |                |        |        | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |       |        |'.format(
            val_metrics['valid_loss'], val_metrics['acc'], val_metrics['precision'], val_metrics['recall'], val_metrics['roc'], val_metrics[best_key] ))

        best_f2 = val_metrics[best_key]

    if args.val:
        return

    model.train()

    train_iter = 0

    for epoch in range(args.start_epoch, args.num_epochs):
        #train_loader, val_loader = get_train_val_loaders(batch_size=args.batch_size, val_batch_size=args.val_batch_size, val_num=args.val_num)

        train_loss = 0

        current_lr = get_lrs(optimizer)  #optimizer.state_dict()['param_groups'][2]['lr']
        bg = time.time()
        for batch_idx, data in enumerate(train_loader):
            train_iter += 1
            img, target, target_aux, weights  = data
            img, target, target_aux, weights = img.cuda(), target.cuda(), target_aux.cuda(), weights.cuda()
            
            if 'bert' in args.model_name:
                output = model(img, attention_mask=(img>0), labels=None)
            else:
                output = model(img)
            output_cls = output[:, :1].squeeze()
            output_aux = output[:, 1:]
            #output = output.squeeze()
            
            loss = criterion(output_cls, output_aux, target, target_aux, weights)
            #loss_aux = _reduce_loss(criterion(output_aux, target_aux.float()))
            batch_size = img.size(0)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            #(batch_size * loss).backward()
            if batch_idx % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()
            print('\r {:4d} | {:.7f} | {:06d}/{} | {:.4f} | {:.4f} |'.format(
                epoch, float(current_lr[0]), args.batch_size*(batch_idx+1), train_loader.num, loss.item(), train_loss/(batch_idx+1)), end='')

            if train_iter > 0 and train_iter % args.iter_val == 0:
                if isinstance(model, DataParallel):
                    torch.save(model.module.state_dict(), model_file+'_latest')
                else:
                    torch.save(model.state_dict(), model_file+'_latest')

                val_metrics = validate(args, model, val_loader)
                
                _save_ckp = ''
                if args.always_save or val_metrics[best_key] > best_f2:
                    best_f2 = val_metrics[best_key]
                    if isinstance(model, DataParallel):
                        torch.save(model.module.state_dict(), model_file)
                    else:
                        torch.save(model.state_dict(), model_file)
                    _save_ckp = '*'
                print(' {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} |  {:4s} |'.format(
                    val_metrics['valid_loss'], val_metrics['acc'], val_metrics['precision'], val_metrics['recall'], val_metrics['roc'], best_f2,
                    (time.time() - bg) / 60, _save_ckp))

                model.train()
                current_lr = get_lrs(optimizer)

    #del model, optimizer, lr_scheduler

def get_lrs(optimizer):
    lrs = []
    if isinstance(optimizer, BertAdam):
        lrs = optimizer.get_lr()
    else:
        for pgs in optimizer.state_dict()['param_groups']:
            lrs.append(pgs['lr'])
    lrs = ['{:.9f}'.format(x) for x in lrs]
    return lrs

def validate(args, model: nn.Module, valid_loader):
    model.eval()
    all_losses, all_scores, all_targets = [], [], []
    with torch.no_grad():
        for inputs, targets, aux_targets, weights in valid_loader:
            all_targets.append(targets)
            inputs, targets, aux_targets, weights = inputs.cuda(), targets.cuda(), aux_targets.cuda(), weights.cuda()
            #outputs, aux_outputs = model(inputs)
            #outputs = outputs.squeeze()
            if 'bert' in args.model_name:
                output = model(inputs, attention_mask=(inputs>0), labels=None)
            else:
                output = model(inputs)
            output_cls = output[:, :1].squeeze()
            output_aux = output[:, 1:]

            loss = criterion(output_cls, output_aux, targets, aux_targets, weights)
            all_losses.append(loss.item())
            scores = torch.sigmoid(output_cls)
            all_scores.append(scores.cpu())

    all_scores = torch.cat(all_scores, 0).numpy()
    all_preds = (all_scores > 0.5).astype(np.int16)
    all_targets = torch.cat(all_targets).numpy().astype(np.int16)

    #print(all_targets)
    #print(all_preds)
    metrics = {}
    metrics['valid_loss'] = np.mean(all_losses)

    tp = ((all_preds == all_targets).astype(np.int16) * all_targets).sum()
    metrics['precision'] = tp / (all_preds.sum() + 1e-6)
    metrics['recall'] = tp / (all_targets.sum() + 1e-6)
    metrics['acc'] = (all_preds == all_targets).sum() / len(all_targets)
    metrics['true'] = all_targets.sum()

    #roc_score = roc_auc_score(all_targets.numpy().astype(np.int32), all_scores.numpy())
    metrics['roc'] = auc_score(all_scores, valid_loader.df)
    #print(metrics)
    return metrics


def pred_model_output(model, loader):
    model.eval()
    outputs = []
    labels = []
    with torch.no_grad():
        for inputs in tqdm(loader, total=loader.num // loader.batch_size):
            inputs = inputs.cuda()
            if 'bert' in args.model_name:
                output = model(inputs, attention_mask=(inputs>0), labels=None)[:, :1].squeeze()
            else:
                output = model(inputs)[:, :1].squeeze()
            outputs.append(torch.sigmoid(output).cpu())
    outputs = torch.cat(outputs).numpy()
    print(outputs.shape)
    return outputs

def predict(args):
    model, _, tokenizer = create_model(args)
    #model = create_model(args)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    test_loader = get_test_loader(batch_size=args.val_batch_size, model_name=args.model_name, tokenizer=tokenizer, clean_text=args.clean_text)
    scores = pred_model_output(model, test_loader)

    print(scores.shape)
    print(scores[:2])

    create_submission(args, scores)

def create_submission(args, scores):
    df = pd.read_csv(os.path.join(settings.DATA_DIR, 'test.csv'))
    df['prediction'] = scores

    df.to_csv(args.sub_file, header=True, index=False, columns=['id', 'prediction'])

def mean_df_old(args):
    df_files = args.mean_df.split(',')
    print(df_files)
    dfs = []
    for fn in df_files:
        dfs.append(pd.read_csv(fn))
    mean_pred = np.mean([dfi.prediction.values for dfi in dfs], 0).astype(np.float32)
    dfs[0].prediction = mean_pred
    dfs[0].to_csv(args.sub_file, index=False, header=True)

def mean_df(args):
    df_files = args.mean_df.split(',')
    print(df_files)
    dfs = []
    for fn in df_files:
        dfs.append(pd.read_csv(fn))
    if args.weights is None:
        w = np.array([1] * len(dfs))
    else:
        w = np.array([int(x) for x in args.weights.split(',')])
    w = w / w.sum()
    print('w:', w)

    assert len(w) == len(dfs)

    df_sub = pd.read_csv(os.path.join(settings.DATA_DIR, 'test.csv'))

    preds = None
    for df, weight in zip(dfs, w):
        if preds is None:
            preds = df.prediction.values * weight
        else:
            preds += df.prediction.values * weight

    df_sub['prediction'] = preds
    df_sub.to_csv(args.sub_file, header=True, index=False, columns=['id', 'prediction'])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Landmark detection')
    parser.add_argument('--run_name', required=True, type=str, help='learning rate')
    parser.add_argument('--model_name', default='bert-base-uncased', type=str, help='learning rate')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='min learning rate')
    parser.add_argument('--batch_size', default=240, type=int, help='batch_size')
    parser.add_argument('--val_batch_size', default=1024, type=int, help='batch_size')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--iter_val', default=400, type=int, help='start epoch')
    parser.add_argument('--num_epochs', default=1, type=int, help='epoch')
    parser.add_argument('--optim_name', default='BertAdam', choices=['SGD', 'Adam', 'BertAdam'], help='optimizer')
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument('--lrs', default='plateau', choices=['cosine', 'plateau'], help='LR sceduler')
    parser.add_argument('--patience', default=6, type=int, help='lr scheduler patience')
    parser.add_argument('--factor', default=0.5, type=float, help='lr scheduler factor')
    parser.add_argument('--t_max', default=8, type=int, help='lr scheduler patience')
    parser.add_argument('--init_ckp', default=None, type=str, help='resume from checkpoint path')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--ckp_name', type=str, default='best_model.pth',help='check point file name')
    parser.add_argument('--sub_file', type=str, default='sub1.csv')
    parser.add_argument('--mean_df', type=str, default=None)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--use_path', action='store_true')
    parser.add_argument('--no_first_val', action='store_true')
    parser.add_argument('--clean_text', action='store_true')
    parser.add_argument('--ifold', default=0, type=int, help='lr scheduler patience')
    parser.add_argument('--always_save',action='store_true', help='alway save')
    parser.add_argument('--val_num', default=50000, type=int, help='number of val data')
    parser.add_argument('--num_classes', default=8, type=int, help='image size')
    parser.add_argument('--convert_model', action='store_true')
    
    args = parser.parse_args()
    print(args)
    #test_model(args)
    #exit(1)

    if args.convert_model:
        convert_model(args)
        print('done')
    elif args.mean_df:
        mean_df(args)
    elif args.predict:
        predict(args)
    else:
        train(args)
