import random
from typing import Sequence, Dict, Tuple, Union

import torch
from torch.utils.data import Dataset


'''
データセットを分割するための2つの排反なインデックス集合を生成する関数
dataset    : 分割対称のデータセット
ratio      : 1つ目のセットに含めるデータ量の割合
random_seed: 分割結果を不変にするためのシード
'''
def generate_subset(dataset: Dataset, ratio: float,
                    random_seed: int=0):
    # サブセットの大きさを計算
    size = int(len(dataset) * ratio)

    indices = list(range(len(dataset)))

    # 二つのセットに分ける前にシャッフル
    random.seed(random_seed)
    random.shuffle(indices)

    # セット1とセット2のサンプルのインデックスに分割
    indices1, indices2 = indices[:size], indices[size:]

    return indices1, indices2
'''
サンプルからミニバッチを生成するcollate関数
batch     : CocoCaptionsからサンプルした複数の画像とラベルをまとめたもの
word_to_id: 単語->単語ID辞書
'''
def collate_func(batch: Sequence[Tuple[Union[torch.Tensor, str]]],
                 word_to_id: Dict[str, int]):
    imgs, captions = zip(*batch)

    # それぞれのサンプルの5個のキャプションの中から1つを選択してトークナイズ
    captions = [tokenize_caption(
        random.choice(cap), word_to_id) for cap in captions]

    # キャプションの長さが降順になるように並び替え
    batch = zip(imgs, captions)
    batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)
    imgs, captions = zip(*batch)
    imgs = torch.stack(imgs)

    lengths = [cap.shape[0] for cap in captions]
    targets = torch.full((len(captions), max(lengths)),
                         word_to_id['<null>'], dtype=torch.int64)
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    return imgs, targets, lengths

def collate_func5(batch: Sequence[Tuple[Union[torch.Tensor, str]]]):
    imgs, targets = zip(*batch)
    #print( "targets:", targets )

    ## それぞれのサンプルの5個のキャプションの中から1つを選択してトークナイズ
    #captions = [tokenize_caption(
    #    random.choice(cap), word_to_id) for cap in captions]

    # キャプションの長さが降順になるように並び替え
    #batch = zip(imgs, captions)
    #batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)
    #imgs, captions = zip(*batch)
    
    result_size = max( imgs[0].shape[1], imgs[0].shape[2] )
    max_height = result_size
    max_width = result_size
    for img in imgs:
        height, width = img.shape[1:]
        max_height = max(max_height, height)
        max_width = max(max_width, width)

    imgs = batch[0][0].new_zeros(
        (len(batch), 3, max_height, max_width))
    # 真偽値を保持するマスクのテンソルは真の値で初期化
    masks = batch[0][0].new_ones(
        (len(batch), max_height, max_width), dtype=torch.bool)
    targets = []
    for i, (img, target) in enumerate(batch):
        height, width = img.shape[1:]
        imgs[i, :, :height, :width] = img
        # 画像領域には偽の値を設定
        masks[i, :height, :width] = False
        targets.append(target)
        #captionss.append(caption)    
    
    #print( "targets:", targets )
    
    #imgs = torch.stack(imgs)
    targets = torch.tensor( targets )

    #lengths = [cap.shape[0] for cap in captions]
    #targets = torch.full((len(captions), max(lengths)),
    #                     word_to_id['<null>'], dtype=torch.int64)
    #for i, cap in enumerate(captions):
    #    end = lengths[i]
    #    targets[i, :end] = cap[:end]
    
    #return imgs, masks, targets, lengths
    return imgs, masks, targets
    
def collate_func4(batch: Sequence[Tuple[Union[torch.Tensor, str]]],
                 word_to_id: Dict[str, int]):
    imgs, captions = zip(*batch)

    # それぞれのサンプルの5個のキャプションの中から1つを選択してトークナイズ
    captions = [tokenize_caption(
        random.choice(cap), word_to_id) for cap in captions]

    # キャプションの長さが降順になるように並び替え
    batch = zip(imgs, captions)
    batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)
    imgs, captions = zip(*batch)
    
    result_size = max( imgs[0].shape[1], imgs[0].shape[2] )
    max_height = result_size
    max_width = result_size
    for img in imgs:
        height, width = img.shape[1:]
        max_height = max(max_height, height)
        max_width = max(max_width, width)

    imgs = batch[0][0].new_zeros(
        (len(batch), 3, max_height, max_width))
    # 真偽値を保持するマスクのテンソルは真の値で初期化
    masks = batch[0][0].new_ones(
        (len(batch), max_height, max_width), dtype=torch.bool)
    targets = []
    for i, (img, caption) in enumerate(batch):
        height, width = img.shape[1:]
        imgs[i, :, :height, :width] = img
        # 画像領域には偽の値を設定
        masks[i, :height, :width] = False

        #captionss.append(caption)    
    
    #imgs = torch.stack(imgs)

    lengths = [cap.shape[0] for cap in captions]
    targets = torch.full((len(captions), max(lengths)),
                         word_to_id['<null>'], dtype=torch.int64)
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    return imgs, masks, targets, lengths

'''
サンプルからミニバッチを生成するcollate関数
batch     : CocoCaptionsからサンプルした複数の画像とラベルをまとめたもの
word_to_id: 単語->単語ID辞書
'''
def collate_func3(batch: Sequence[Tuple[Union[torch.Tensor, str]]],
                 word_to_id: Dict[str, int]):
    imgs, captions = zip(*batch)

    # それぞれのサンプルの5個のキャプションの中から1つを選択してトークナイズ
    captions = [tokenize_caption(
        random.choice(cap), word_to_id) for cap in captions]

    # キャプションの長さが降順になるように並び替え
    batch = zip(imgs, captions)
    batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)
    imgs, captions = zip(*batch)
    
    max_height = 0
    max_width = 0
    for img, _ in batch:
        height, width = img.shape[1:]
        max_height = max(max_height, height)
        max_width = max(max_width, width)
        
    const =  max( max_height, max_width )

    imgs2 = batch[0][0].new_zeros( (len(batch), 3, const, const ))
    masks = batch[0][0].new_ones( ( len(batch), const, const ), dtype=torch.bool )
    
    for i, img in enumerate( imgs ):
    	height, width = img.shape[1:]
    	#print( "height:", height )
    	#print( "width:", width )
    	imgs2[i, :, :height, :width ] = img
    	masks[i, :height, :width] = False
    #imgs2 = torch.stack(imgs2)
    imgs2 = torch.Tensor( imgs2 )

    lengths = [cap.shape[0] for cap in captions]
    targets = torch.full((len(captions), max(lengths)),
                         word_to_id['<null>'], dtype=torch.int64)
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    return imgs2, targets, lengths, masks
    
def collate_func2(batch: Sequence[Tuple[Union[torch.Tensor, str]]],
                 word_to_id: Dict[str, int]):
    imgs, captions = zip(*batch)

    # それぞれのサンプルの5個のキャプションの中から1つを選択してトークナイズ
    captions = [tokenize_caption2(
        random.choice(cap), word_to_id) for cap in captions]

    # キャプションの長さが降順になるように並び替え
    batch = zip(imgs, captions)
    batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)
    imgs, captions = zip(*batch)
    imgs = torch.stack(imgs)

    lengths = [cap.shape[0] for cap in captions]
    targets = torch.full((len(captions), max(lengths)),
                         word_to_id['<null>'], dtype=torch.int64)
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    return imgs, targets, lengths


'''
トークナイザ - 文章(caption)を単語IDのリスト(tokens_id)に変換
caption   : 画像キャプション
word_to_id: 単語->単語ID辞書
'''
def tokenize_caption(caption: str, word_to_id: Dict[str, int]):
    tokens = caption.lower().split()
    
    tokens_temp = []    
    # 単語についたピリオド、カンマを削除
    for token in tokens:
        if token == '.' or token == ',':
            continue

        token = token.rstrip('.')
        token = token.rstrip(',')
        
        tokens_temp.append(token)
    
    tokens = tokens_temp        
        
    # 文章(caption)を単語IDのリスト(tokens_id)に変換
    tokens_ext = ['<start>'] + tokens + ['<end>']
    tokens_id = []
    for k in tokens_ext:
        if k in word_to_id:
            #if word_to_id[k] == 0:
            #    print( "token_id is 0 " )
            #if k == '<start>':
            #    print( "word_to_id['<start>']:", word_to_id['<start>'] )
            #if k == '<end>':
            #    print( "word_to_id['<end>']:", word_to_id['<end>'] )
            tokens_id.append(word_to_id[k])
        else:
            #print( "<unk>" )
            #print( "word_to_id['<unk>']", word_to_id['<unk>'] )
            tokens_id.append(word_to_id['<unk>'])
    
    return torch.Tensor(tokens_id)
    
def tokenize_caption2(caption: str, word_to_id: Dict[str, int]):
    tokens = caption.lower().split()
    
    tokens_temp = []    
    # 単語についたピリオド、カンマを削除
    for token in tokens:
        if token == '.' or token == ',':
            continue

        token = token.rstrip('.')
        token = token.rstrip(',')
        
        tokens_temp.append(token)
    
    tokens = tokens_temp        
        
    # 文章(caption)を単語IDのリスト(tokens_id)に変換
    #tokens_ext = ['<start>'] + tokens + ['<end>']
    tokens_ext = tokens
    tokens_id = []
    for k in tokens_ext:
        if k in word_to_id:
            tokens_id.append(word_to_id[k])
        else:
            tokens_id.append(word_to_id['<unk>'])
    
    return torch.Tensor(tokens_id)
