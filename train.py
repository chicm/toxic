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
from models import ToxicModel, create_model
from torch.nn import DataParallel
from tqdm import tqdm


MODEL_DIR = settings.MODEL_DIR



#criterion = nn.BCEWithLogitsLoss(weight=cls_weights, reduction='none')
criterion = nn.BCEWithLogitsLoss(reduction='none')
#criterion = FocalLoss2()



def train(args):
    print('start training...')
    model, model_file = create_model(args)
    #model = model.cuda()
    

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

    if args.lrs == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, args.t_max, eta_min=args.min_lr)
    #ExponentialLR(optimizer, 0.9, last_epoch=-1) #CosineAnnealingLR(optimizer, 15, 1e-7) 

    _, val_loader = get_train_val_loaders(batch_size=args.batch_size, val_batch_size=args.val_batch_size, val_num=args.val_num)

    best_f2 = 999.
    best_key = 'valid_loss'

    print('epoch |    lr    |      %        |  loss  |  avg   |  loss  |  acc  |   best  | time |  save |')

    if not args.no_first_val:
        val_metrics = validate(args, model, val_loader)
        print('val   |          |               |        |        | {:.4f} | {:.4f} | {:.4f} |        |        |'.format(
            val_metrics['valid_loss'], val_metrics['acc'], val_metrics['acc'], val_metrics[best_key] ))

        best_f2 = val_metrics[best_key]

    if args.val:
        return

    model.train()

    if args.lrs == 'plateau':
        lr_scheduler.step(best_f2)
    else:
        lr_scheduler.step()
    train_iter = 0

    for epoch in range(args.start_epoch, args.epochs):
        train_loader, val_loader = get_train_val_loaders(batch_size=args.batch_size, val_batch_size=args.val_batch_size, val_num=args.val_num)

        train_loss = 0

        current_lr = get_lrs(optimizer)  #optimizer.state_dict()['param_groups'][2]['lr']
        bg = time.time()
        for batch_idx, data in enumerate(train_loader):
            train_iter += 1
            img, target  = data
            img, target = img.cuda(), target.cuda()
            
            output = model(img).squeeze()
            
            #loss = criterion(output, target)
            #loss.backward()

            loss = _reduce_loss(criterion(output, target.float()))
            batch_size = img.size(0)
            (batch_size * loss).backward()

            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            print('\r {:4d} | {:.6f} | {:06d}/{} | {:.4f} | {:.4f} |'.format(
                epoch, float(current_lr[0]), args.batch_size*(batch_idx+1), train_loader.num, loss.item(), train_loss/(batch_idx+1)), end='')

            if train_iter > 0 and train_iter % args.iter_val == 0:
                if isinstance(model, DataParallel):
                    torch.save(model.module.state_dict(), model_file+'_latest')
                else:
                    torch.save(model.state_dict(), model_file+'_latest')

                val_metrics = validate(args, model, val_loader)
                
                _save_ckp = ''
                if args.always_save or val_metrics[best_key] < best_f2:
                    best_f2 = val_metrics[best_key]
                    if isinstance(model, DataParallel):
                        torch.save(model.module.state_dict(), model_file)
                    else:
                        torch.save(model.state_dict(), model_file)
                    _save_ckp = '*'
                print(' {:.4f} | {:.4f} | {:.4f} | {:.2f} |  {:4s} |'.format(
                    val_metrics['valid_loss'], val_metrics['acc'], best_f2,
                    (time.time() - bg) / 60, _save_ckp))

                model.train()
                
                if args.lrs == 'plateau':
                    lr_scheduler.step(best_f2)
                else:
                    lr_scheduler.step()
                current_lr = get_lrs(optimizer)

    #del model, optimizer, lr_scheduler
        
def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs


def validate(args, model: nn.Module, valid_loader):
    model.eval()
    all_losses, all_scores, all_targets = [], [], []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            all_targets.append(targets)
            #if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets.float())
            all_losses.append(_reduce_loss(loss).item())
            #all_losses.append(loss.item())
            scores = torch.sigmoid(outputs)
            all_scores.append(scores.cpu())

    all_scores = torch.cat(all_scores, 0)
    all_preds = (all_scores > 0.5)
    all_targets = torch.cat(all_targets)

    #print(all_targets)
    #print(all_preds)

    acc = (all_preds == all_targets.byte()).sum().item() / len(all_targets)

    metrics = {}
    metrics['valid_loss'] = np.mean(all_losses)
    metrics['acc'] = acc
    print(metrics)
    return metrics

def _reduce_loss(loss):
    #print('loss shape:', loss.shape)
    return loss.sum() / loss.shape[0]

def pred_model_output(model, loader, labeled=True):
    model.eval()
    outputs = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(loader, total=loader.num // loader.batch_size):
            if labeled:
                img = batch[0].cuda()
                labels.append(batch[1])
            else:
                img = batch.cuda()
            output = model(img)
            outputs.append(torch.sigmoid(output).cpu())
            #outputs.append(torch.softmax(output, 1).cpu())
    outputs = torch.cat(outputs).numpy()
    print(outputs.shape)
    if labeled:
        labels = torch.cat(labels).numpy()
        return outputs, labels
    else:
        return outputs

def find_val_threshold(args):
    labels = np.load('output/val/val_labels.npy')
    preds = []
    for i in range(args.tta_num):
        preds.append(np.load('output/val/val_tta_pred_{}.npy'.format(i)))
    tta_pred = np.mean(preds, 0)
    print(tta_pred.shape, labels.shape)

    argsorted = tta_pred.argsort(axis=1)

    best_f2 = 0.
    for threshold in [0.0005, 0.005, 0.01, 0.05, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.15, 0.2]:
        bin_pred = binarize_prediction(tta_pred, threshold, argsorted)
        #print(bin_pred[:2])
        f2 = get_f2_score(bin_pred, labels)
        print('threshold: {}, f2: {}'.format(threshold, f2))
        best_f2 = max(best_f2, f2)
    thresholds = [0.08] * settings.N_CLASSES

    #best_th = thresholds
    #for i in range(100):
    for i in tqdm(top200_cls, total=len(top200_cls)):
        for j in np.arange(5, 25) / 100:
            th = thresholds.copy()
            th[i] = j
            bin_pred = binarize_prediction(tta_pred, th, argsorted)
            #print(bin_pred[:2])
            f2 = get_f2_score(bin_pred, labels)
            if f2 > best_f2:
                print('best: {:.6f}, class_id: {}, th: {:.4f}'.format(f2, i, j))
                best_f2 = f2
                thresholds[i] = j
    np.save('output/val/best_val_th.npy', np.array(thresholds))

    tmp = np.load('output/val/best_val_th.npy')
    print(tmp)


def predict(args):
    model, _ = create_model(args)
    #model = model.cuda()
    if torch.cuda.device_count() > 1:
        model_name = model.name
        model = DataParallel(model)
        model.name = model_name
    model = model.cuda()

    preds = []
    for i in range(args.tta_num):
        test_loader = get_test_loader(val_batch_size=args.val_batch_size, tta=i, dev_mode=args.dev_mode)
        pred = pred_model_output(model, test_loader, labeled=False)
        #print(pred[:2])
        preds.append(pred)
    tta_pred = np.mean(preds, 0)

    print(tta_pred.shape)
    print(tta_pred[:2])

    df = pd.DataFrame(
        data=tta_pred,
        index=test_loader.ids,
        columns=map(str, range(settings.N_CLASSES)))
    print(df.head())
    df.to_csv(args.backbone+'_pred_score.csv', index=True, index_label='id')

    create_submission(args, [args.backbone+'_pred_score.csv'], args.threshold)

def get_classes(item):
    return ' '.join(cls for cls, is_present in item.items() if is_present)

def create_submission(args, score_files, thresholds):
    print('thresholds:', thresholds)
    print('score_files:', score_files)
    df = pd.read_csv(score_files[0], index_col='id')

    preds = []
    for score_file in score_files:
        dfi = pd.read_csv(score_file, index_col='id')
        preds.append(dfi.values)
    mean_preds = np.mean(preds, 0)

    df[:] = binarize_prediction(mean_preds, threshold=thresholds)
    df = df.apply(get_classes, axis=1)
    df.name = 'attribute_ids'
    df.to_csv(args.sub_file, header=True)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Landmark detection')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=0.0001, type=float, help='min learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
    parser.add_argument('--val_batch_size', default=512, type=int, help='batch_size')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--iter_val', default=200, type=int, help='start epoch')
    parser.add_argument('--epochs', default=200, type=int, help='epoch')
    parser.add_argument('--optim', default='SGD', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--lrs', default='plateau', choices=['cosine', 'plateau'], help='LR sceduler')
    parser.add_argument('--patience', default=6, type=int, help='lr scheduler patience')
    parser.add_argument('--factor', default=0.5, type=float, help='lr scheduler factor')
    parser.add_argument('--t_max', default=8, type=int, help='lr scheduler patience')
    parser.add_argument('--init_ckp', default=None, type=str, help='resume from checkpoint path')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--ckp_name', type=str, default='best_model.pth',help='check point file name')
    parser.add_argument('--sub_file', type=str, default='sub1.csv')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--no_first_val', action='store_true')
    parser.add_argument('--always_save',action='store_true', help='alway save')
    parser.add_argument('--val_num', default=10000, type=int, help='number of val data')
    #parser.add_argument('--img_sz', default=256, type=int, help='image size')
    
    args = parser.parse_args()
    print(args)
    #test_model(args)
    #exit(1)

    train(args)
