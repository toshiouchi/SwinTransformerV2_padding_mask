import random
from PIL import Image
from typing import Sequence, Callable

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


class Resize2:
    '''
    画像をアスペクト比を保持してリサイズするクラス
    Resize した結果 image の長辺の長さ
    '''
    def __init__(self, result_size: int):
        self.result_size = result_size

    '''
    元の画像を長辺が self.result_size になるように画像をリサイズする関数
    img   : リサイズする画像
    '''
    def __call__(self, img: Image):
        # width が長辺か、height が長辺か調べる。
        width, height = img.size
        max_size = max( width, height )
        w_flag = False
        h_flag = False
        if max_size == width:
            w_flag = True
        else:
            h_flag = True
        
        # resize する weidth と height を決定する
        if w_flag == True:
           r_ratio = self.result_size / width
           resized_width = self.result_size
           resized_height = int( height * r_ratio )  
        else:
           r_ratio = self.result_size / height
           resized_height = self.result_size
           resized_width = int( width * r_ratio )


        # 指定した大きさに画像をリサイズ
        img = F.resize(img, (resized_height, resized_width))

        return img
