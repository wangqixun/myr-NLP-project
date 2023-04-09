import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
import torchvision.transforms as transforms
# from transformers import AutoModel
# from transformers import AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import random
import os
import copy
import pandas as pd


class MyDataset(Dataset):
    def __init__(
        self, 
        ann_file, 
        cfg, 
        mode='tra', 
    ):
        super(MyDataset, self).__init__()

        data = np.array(pd.read_csv(ann_file))
        self.data = data
        self.mode = mode
        self.cfg = cfg

    def __getitem__(self, index):
        if self.mode == 'test':
            d = self.data[index]
            context = d[1]
            return context
        else :
            d = self.data[index]
            context, label = d
            label = int(label)
            return context, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    d = '/share/wangqixun/workspace/bs/myr/data/val.csv'
    D = MyDataset(d, cfg={})
    nb_1 = 0
    for i, d in enumerate(D):
        _, l = d
        if l==1:
            nb_1 += 1
    print(nb_1/len(D))


    np.random.seed(666)

    ann_file1 = '/share/wangqixun/workspace/bs/myr/data/test_samples.csv'
    ann_file2 = '/share/wangqixun/workspace/bs/myr/data/train_samples.csv'
    data1 = pd.read_csv(ann_file1)
    data2 = pd.read_csv(ann_file2)
    data = pd.concat([data1, data2])

    data = np.array(data)
    np.random.shuffle(data)

    data_tra = data[:int(len(data)*0.7)]
    data_val = data[int(len(data)*0.7):]

    data_tra = pd.DataFrame(data_tra, columns=['content', 'label'])
    data_val = pd.DataFrame(data_val, columns=['content', 'label'])

    data_tra.to_csv('/share/wangqixun/workspace/bs/myr/data/tra.csv', index=False)
    data_val.to_csv('/share/wangqixun/workspace/bs/myr/data/val.csv', index=False)






