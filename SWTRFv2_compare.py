from collections import deque
import copy
from tqdm import tqdm
#from tqdm.notebook import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as T
import transform as T2
from torchvision import datasets, transforms
from torch.utils import data as data_utils
from torch.utils.data.dataset import Subset
import glob
from PIL import Image
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup
from torch.nn.init import xavier_uniform_, constant_, normal_
#import timm
from torch import Tensor

import sys
import os

import util
import eval

from SwinTransformerV2 import SwinTransformerV2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class MyDatasets(torch.utils.data.Dataset):
    def __init__(self, mode):

        traindir = os.path.join("/mnt/ssd2/imagenet", 'train')     # /train/ を指定されたパスに追加
        valdir = os.path.join("/mnt/ssd2/imagenet", 'validation_folder')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])     # 正規化定数        
        
        if mode == "train":
            self.dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                transforms.Resize(size=(224, 224)),
                #transforms.RandomResizedCrop(224),      # 画像をサイズ224に切り出しもしくはリサイズ
                #T2.Resize2(224),
                transforms.RandomHorizontalFlip(),      # ランダムに画像をフリップ（水増し）
                transforms.ToTensor(),
                normalize,
            ]))
        else:
            self.dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                transforms.Resize(size=(224, 224)),
                #transforms.RandomResizedCrop(224),      # 画像をサイズ224に切り出しもしくはリサイズ
                #T2.Resize2(224),
                transforms.RandomHorizontalFlip(),      # ランダムに画像をフリップ（水増し）
                transforms.ToTensor(),
                normalize,
            ]))

        self.datanum = len(self.dataset) 

        self.mode = mode
            
    def __len__(self):
        #return self.datanum
        #return self.datanum // 1000
        #return self.datanum // 10
        return self.datanum // 100
    
    def __getitem__(self, idx):
        batch = self.dataset[idx]
        
        return batch
        

train_dataset = MyDatasets('train')
val_dataset = MyDatasets('val')
print( "3")

num_workers = 0 if device==torch.device('cpu') else 4

print( "num_workers:", num_workers )

collate_func_lambda = lambda x : util.collate_func5(x)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=8,
                                          shuffle=True,
                                          num_workers=num_workers,)
                                          #collate_fn=collate_func_lambda)

print( "len train_loader:", len(train_loader))

val_loader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=8,
                                          shuffle=False,
                                          num_workers=num_workers,)
                                          #collate_fn=collate_func_lambda)

print( "len val_loader:", len(val_loader))

model = SwinTransformerV2( )

class Config:
    
    #ハイパーパラメータとオプションの設定
    
    def __init__(self):
        self.val_ratio = 0.1       # 検証に使う学習セット内のデータの割合
        #self.train_ratio = 0.9
        self.num_epochs = 25       # 学習エポック数
        self.lr = 1e-3             # 学習率
        self.moving_avg = 20       # 移動平均で計算する損失と正確度の値の数
        self.device = 'cuda'       # 学習に使うデバイス
        #self.device = 'cpu'       # 学習に使うデバイス

import ssl

config = Config()

ssl._create_default_https_context = ssl._create_unverified_context

print(f'学習セットのサンプル数　: {len(train_loader)}')
print(f'検証セットのサンプル数　: {len(val_loader)}')


    
# 目的関数の生成
loss_func = F.cross_entropy
#loss_func = nn.CrossEntropyLoss()

# 検証セットの結果による最良モデルの保存用変数
val_loss_best = float('inf')
model_best = None


# モデルを指定デバイスに転送(デフォルトはGPU)
model.to(config.device)

# 最適化器の生成
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

num_warmup_steps = len( train_loader ) * config.num_epochs * 0.1
num_train_steps = len( train_loader ) * config.num_epochs

scheduler = get_linear_schedule_with_warmup( optimizer, num_warmup_steps, num_train_steps )

PATH = 'model_compare_best.pth'
use_save_param = False
if os.path.isfile(PATH) and use_save_param:
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    
    epoch_begin = checkpoint['epoch']
    loss = checkpoint['loss']
    print( "loaded parameters." )
else:
    epoch_begin = 0

global_step = 0    
global_val_step = 0

f_train = open( "P_M_compare_train.log", mode="w", encoding = "UTF-8" )
f_train.close()
f_val = open( "P_M_compare_val.log", mode="w", encoding = "UTF-8" )
f_val.close()

f_train = open( "P_M_compare_train.log", mode="w", encoding = "UTF-8" )
f_val = open( "P_M_compare_val.log", mode="w", encoding = "UTF-8" ) 
 
    
for epoch in range(epoch_begin, config.num_epochs):
    model.train()

    with tqdm(train_loader) as pbar:
    #with tqdm(val_loader) as pbar:
        pbar.set_description(f'[エポック {epoch + 1}]')

        # 移動平均計算用
        losses = deque()
        accs = deque()
        #for x, masks, y in pbar:
        for x, y in pbar:
            global_step += 1
            # データをモデルと同じデバイスに転送
            x = x.to(device)
            #masks = masks.to(device)
            y = y.to(device)

            # パラメータの勾配をリセット
            optimizer.zero_grad()

            # 順伝播
            #y_pred  = model(x, masks)
            y_pred  = model(x)

            # 学習データに対する損失と正確度を計算
            #print( "y_pred:", y_pred )
            #print( "size y_pred:", y_pred.size() )
            #print( "y size:", y.size())
            #print( "y:", y )
            loss = loss_func(y_pred, y)
            accuracy = (y_pred.argmax(dim=1) == \
                        y).float().mean()

            # 誤差逆伝播
            loss.backward()

            # パラメータの更新
            optimizer.step()
            scheduler.step()

            # 移動平均を計算して表示
            losses.append(loss.item())
            accs.append(accuracy.item())
            if len(losses) > config.moving_avg:
                losses.popleft()
                accs.popleft()
            pbar.set_postfix({
                'loss': torch.Tensor(losses).mean().item(),
                'accuracy': torch.Tensor(accs).mean().item()})

            f_train.write( "global_step:" + str( global_step ) + ",loss:" + str(torch.Tensor(losses).mean().item()) + ",acc:" + str( torch.Tensor(accs).mean().item() ) + "\n" )
            f_train.flush()

    # 検証セットを使って精度評価
    #val_loss, val_accuracy = eval.evaluate3(
    val_loss, val_accuracy, global_val_step = eval.evaluate4(
        val_loader, model, loss_func, device, global_val_step, f_val )
    print(f'検証　: loss = {val_loss:.3f}, '
            f'accuracy = {val_accuracy:.3f}')

    # より良い検証結果が得られた場合、モデルを記録
    if val_loss < val_loss_best:
        val_loss_best = val_loss
        #model_best = model.copy()
        model_best = copy.deepcopy(model)
        torch.save(
            {'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,},
            f'model_compare_best.pth')

f_train.close()
f_val.close()

torch.save(
    {'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,},
    f'model_compare.pth')


