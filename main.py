import os
import random
import argparse
import logging
import torch
import numpy as np
import pandas as pd
import torch.nn as nn 
import torch.optim as optim
from datetime import datetime
from sklearn.model_selection import KFold
from utils import get_loader
from model import TCM_GNN
from multi_site_loss import HSIC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device, use_site_loss, site_num, kernel, beta):
    epoch_loss = []
    model.train()
    for batch_graph, labels, pool_ids, sites in train_loader:
        optimizer.zero_grad()
        batch_graph = batch_graph.to(device)
        pool_ids = pool_ids.to(device)
        labels = labels.squeeze().to(device)
        preds, feats = model(batch_graph, pool_ids)
        if use_site_loss:
            loss1 = beta * HSIC(feats, sites, site_num, device, kernel)
            loss2 = loss_fn(preds, labels)
            loss = loss1 + loss2
        else:
            loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()               
        epoch_loss.append(loss.item())
    scheduler.step()   
    return np.mean(epoch_loss)

def val_epoch(model, val_loader, epoch, device):
    model.eval()
    with torch.no_grad():
        all_preds_pb, all_labels = [], []
        for batch_graph, labels, pool_ids, _ in val_loader:
            batch_graph = batch_graph.to(device)
            pool_ids = pool_ids.to(device)
            preds, _ = model(batch_graph, pool_ids)
            preds = preds.cpu()
            preds = torch.sigmoid(preds)

            all_preds_pb.extend(preds.numpy())
            all_labels.extend(labels.numpy())

        num_true = int(sum(all_labels))
        sorted_pred = sorted(all_preds_pb)
        threshold = sorted_pred[-num_true]
        all_preds = [int(pred > threshold) for pred in all_preds_pb]
        all_labels = [int(label) for label in all_labels]

        acc = accuracy_score(all_labels, all_preds)
        pre = precision_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_preds_pb)

        logging.info(f'{epoch:02}: Val acc: {acc:.4f} | Val pre: {pre:.4f} | Val rec: {rec:.4f} | Val f1: {f1:.4f} | Val auc: {auc:.4f}')

def test_epoch(model, test_loader, epoch, device):
    model.eval()
    with torch.no_grad():
        all_preds_pb, all_labels = [], []
        for batch_graph, labels, pool_ids, _ in test_loader:
            batch_graph = batch_graph.to(device)
            pool_ids = pool_ids.to(device)
            preds, _ = model(batch_graph, pool_ids)
            preds = preds.cpu()
            preds = torch.sigmoid(preds)

            all_preds_pb.extend(preds.numpy())
            all_labels.extend(labels.numpy())

        num_true = int(sum(all_labels))
        sorted_pred = sorted(all_preds_pb)
        threshold = sorted_pred[-num_true]
        all_preds = [int(pred > threshold) for pred in all_preds_pb]
        all_labels = [int(label) for label in all_labels]

        acc = accuracy_score(all_labels, all_preds)
        pre = precision_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_preds_pb)
    logging.info(f'{epoch:02}: Test acc: {acc:.4f} | Test pre: {pre:.4f} | Test rec: {rec:.4f} | Test f1: {f1:.4f} | Test auc: {auc:.4f}')

    return acc, pre, rec, f1, auc

def main(args):
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False

        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    modals = ['t1', 't1c', 't2', 'flair']
    used_modals = [modals[int(i)] for i in args.used_modals]
    print(f'Used modals: {used_modals}')
    if args.temp == -1:
        temp = None
    else:
        temp = args.temp

    bool_values = [args.use_site_loss,  args.use_segpool, args.use_attention, args.use_co] 
    ablation_str = ''.join(['1' if b else '0' for b in bool_values])

    log_dir = os.path.join(args.log_path, ablation_str+'_modal_'+args.used_modals)
    model_dir = os.path.join(args.save_path, ablation_str+'_modal_'+args.used_modals)
    os.makedirs(log_dir, exist_ok=True)   
    os.makedirs(model_dir, exist_ok=True)   

    now_date = datetime.now()
    now_date = now_date.strftime("%Y%m%d_%H%M%S") 
    log_fname = f'{log_dir}/{now_date}_seed{args.seed}.out'   
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        filename=log_fname,
        filemode='a'
    )
    logging.info(args) 

    cv_splits = list(KFold(args.num_fold, shuffle=True, random_state=args.seed).split(np.arange(args.len_dataset)))

    all_acc, all_pre, all_rec, all_f1, all_auc = [], [], [], [], []
    for fold in range(args.num_fold):
        print(f'We are training fold {fold}')
        logging.info(f'We are training fold {fold}')
        logging.info('-'*30)
        train_id = cv_splits[fold][0]
        test_id = cv_splits[fold][1]

        train_loader, val_loader, test_loader = get_loader(args, train_id, test_id)

        model = TCM_GNN(relations=used_modals, num_layer=args.num_layer, in_dim=len(used_modals)*args.single_modal_dim, 
                         embed_dim=args.embed_dim, temp=temp, dropout=args.dropout, use_co=args.use_co, 
                         use_attention=args.use_attention, use_segpool=args.use_segpool).to(device)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6) 
        train_loss_track = []
        best_acc, best_pre, best_rec, best_f1, best_auc = 0, 0, 0, 0, 0
        for epoch in range(args.epochs):
            epoch_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device, 
                                     args.use_site_loss, args.site_num, args.kernel, args.beta)
            train_loss_track.append(epoch_loss)

            val_epoch(model, val_loader, epoch, device)
                
            acc, pre, rec, f1, auc = test_epoch(model, test_loader, epoch, device)
            if auc > best_auc:
                best_acc, best_pre, best_rec, best_f1, best_auc = acc, pre, rec, f1, auc
                torch.save(model.state_dict(), f'{model_dir}/seed_{args.seed}_fold{fold}.pt')

        all_acc.append(best_acc)
        all_pre.append(best_pre)
        all_rec.append(best_rec)
        all_f1.append(best_f1)
        all_auc.append(best_auc)

        with open(log_fname, 'a') as f:
            f.write(
                '\n'.join(
                    ('-' * 25,
                     f'Loss tracks:',
                     f'Train loss: {train_loss_track}',
                     f'Best acc/pre/rec/f1/auc: {best_acc, best_pre, best_rec, best_f1, best_auc}\n'
                    )
                )
            )

    with open(log_fname, 'a') as f:
        f.write(
            '\n'.join(
                ('-' * 25,
                 'Test Metrics: Mean and St Dev',
                 f'Accuracy: {np.mean(all_acc):.4f} | {np.std(all_acc):.4f} | {all_acc}',
                 f'Precision: {np.mean(all_pre):.4f} | {np.std(all_pre):.4f} | {all_pre}',
                 f'Recall: {np.mean(all_rec):.4f} | {np.std(all_rec):.4f} | {all_rec}',
                 f'F1: {np.mean(all_f1):.4f} | {np.std(all_f1):.4f} | {all_f1}',
                 f'AUC: {np.mean(all_auc):.4f} | {np.std(all_auc):.4f} | {all_auc}',
                 )
            )
        )



                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Random Seed.') 
    parser.add_argument('--num_fold', type=int, default=5,
                        help='Number of Fold.') 
    parser.add_argument('--used_modals', type=str, default='123',
                        help='Used modals.') 
    parser.add_argument('--use_co', action='store_true') 
    parser.add_argument('--use_attention', action='store_true') 
    parser.add_argument('--use_segpool', action='store_true') 
    parser.add_argument('--use_site_loss', type=bool, default=False) 
    parser.add_argument('--site_num', type=int, default=5) 
    parser.add_argument('--kernel', type=str, default='linear') 
    parser.add_argument('--beta', type=float, default=0.01) 
    parser.add_argument('--data_path', type=str, 
                        help='Path of the Data.') 
    parser.add_argument('--save_path', type=str, 
                        help='Path to Save the Model.')
    parser.add_argument('--log_path', type=str, 
                        help='Path to Save the Log File.') 
    parser.add_argument('--len_dataset', type=int, 
                        help='Length of dataset.') 
    parser.add_argument('--batch_size', type=int, default=48,
                        help='Batch size.') 
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers.') 
    parser.add_argument('--num_layer', type=int, default=2,
                        help='Number of layer.') 
    parser.add_argument('--single_modal_dim', type=int, default=14,
                        help='In dimension.') 
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Embedding dimension.') 
    parser.add_argument('--temp', type=float, default=-1,
                        help='Temp for gumbel_softmax.') 
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate.') 
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum limit on training epochs.') 
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for optimizer.') 
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='L2 regularization penalty.')

    args = parser.parse_args()

    main(args)
