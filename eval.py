from typing import Callable

import torch
from torch import nn
from torch.utils.data import Dataset


'''
data_loader: 評価に使うデータを読み込むデータローダ
model      : 評価対象のモデル
loss_func  : 目的関数
'''
    
def evaluate3(data_loader: Dataset, model: nn.Module,
             loss_func: Callable, device, global_val_step, f_val):
    model.eval()

    losses = []
    preds = []
    for x, m, y in data_loader:
        global_val_step += 1
        with torch.no_grad():
            x = x.to(device)
            m = m.to(device)
            y = y.to(device)

            y_pred = model(x, m )

            loss0 = loss_func(y_pred, y, reduction='none')
            loss1 = torch.mean( loss0 )
            preds0 = y_pred.argmax(dim=1) == y
            preds1 = torch.mean( preds0.float() )
            f_val.write( "global_val_step:" + str( global_val_step ) + ",loss:" + str(loss1.item()) + ",acc:" + str(preds1.item()) + "\n" )
            f_val.flush()
            losses.append(loss0)
            preds.append(preds0)

    loss = torch.cat(losses).mean()
    accuracy = torch.cat(preds).float().mean()

    return loss, accuracy, global_val_step
    
def evaluate4(data_loader: Dataset, model: nn.Module,
             loss_func: Callable, device, global_val_step, f_val):
    model.eval()

    losses = []
    preds = []
    for x, y in data_loader:
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss0 = loss_func(y_pred, y, reduction='none')
            loss1 = torch.mean( loss0 )
            preds0 = y_pred.argmax(dim=1) == y
            preds1 = torch.mean( preds0.float() )
            f_val.write( "global_val_step:" + str( global_val_step ) + ",loss:" + str(loss1.item()) + ",acc:" + str(preds1.item()) + "\n" )
            f_val.flush()
            losses.append(loss0)
            preds.append(preds0)

    loss = torch.cat(losses).mean()
    accuracy = torch.cat(preds).float().mean()

    return loss, accuracy, global_val_step
