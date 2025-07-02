import random
from typing import Sequence, Dict, Tuple, Union

import torch
from torch.utils.data import Dataset

def collate_func5(batch: Sequence[Tuple[Union[torch.Tensor, str]]]):
    imgs, targets = zip(*batch)

    # 与えられた image の長辺の長さに画像サイズ max_height, max_width を合わせる
    result_size = max( imgs[0].shape[1], imgs[0].shape[2] )
    max_height = result_size
    max_width = result_size
    for img in imgs:
        height, width = img.shape[1:]
        max_height = max(max_height, height)
        max_width = max(max_width, width)

    # (batch数、channel=3, max_height, max_width) で全ピクセル 0 の画像を確保。
    imgs = batch[0][0].new_zeros(
        (len(batch), 3, max_height, max_width))
    # (batch数、max_height, max_width) で全値 True のブールでマスクを確保
    masks = batch[0][0].new_ones(
        (len(batch), max_height, max_width), dtype=torch.bool)
    targets = []
    for i, (img, target) in enumerate(batch):
        #実際のimage を画像リストに代入
        height, width = img.shape[1:]
        imgs[i, :, :height, :width] = img
        # マスクの画像領域には偽の値を設定
        masks[i, :height, :width] = False
        # target ラベルを追加
        targets.append(target)
    
    # target ラベルを torch.tensor 化。
    targets = torch.tensor( targets )

    return imgs, masks, targets
