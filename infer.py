import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
import torchvision.transforms as transforms
from transformers import AutoModel
from transformers import AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
import random
import numpy as np
from collections import OrderedDict
from rich import print
import time
import cv2
from glob import glob
import string
from transformers import AdamW, get_linear_schedule_with_warmup

from models_wqx import get_model
from dataset_wqx import MyDataset
from utils_wqx import save_checkpoint, AverageMeter, ProgressMeter


def test_epoch(model, epoch, dataloader, tokenizer):
    print(f"\n\n=> val")
    data_time = AverageMeter('- data', ':4.3f')
    batch_time = AverageMeter('- batch', ':6.3f')
    progress = ProgressMeter(
        len(dataloader), data_time, batch_time, prefix=f"Epoch: [{epoch}]")

    end = time.time()
    model.eval()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    predictions = []

    for batch_index, data_batch in enumerate(tqdm(dataloader)):
        context_str_batch = data_batch

        # data tokenizer
        context_token_batch = tokenizer(context_str_batch, padding=True, truncation=True, max_length=500, return_tensors='pt')
        
        # to gpu
        context_token_batch = {k:v.to(device) for k,v in context_token_batch.items()}

        # forward
        data_input_batch = context_token_batch
        output_batch = model(**data_input_batch)

        pred_batch = output_batch.softmax(dim=-1)
        pred = torch.argmax(pred_batch, dim=-1)
        predictions.extend(pred.cpu().numpy())

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_index % 50 == 0:
            progress.print(batch_index)

    return predictions


def infer20221212():
    checkpoint_file = '/share/wangqixun/workspace/bs/myr/output/v3_macbert/checkpoint_epoch019_acc0.9333.pth.tar'
    output_file = '/share/wangqixun/workspace/bs/myr/output/submit/v3.csv'
    pretrained_transformers = '/share/wangqixun/workspace/github_project/transformers_checkpoint/hfl/chinese-macbert-base'

    cache_dir = '/share/wangqixun/workspace/bs/myr/output/tmp'
    ann_file_test = '/share/wangqixun/workspace/bs/myr/data/all-22-11-30.csv'

    model_cfg = dict(
        pretrained_transformers=pretrained_transformers,
        cache_dir=cache_dir,
    )
    # 模型
    model_dict = get_model(model_cfg, mode='base')
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    print(model)


    data_loader_cfg = {}
    test_dataset = MyDataset(ann_file_test, data_loader_cfg, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4, pin_memory=True)

    # resume
    assert checkpoint_file is not None and os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
    print(f"=> Resume: loaded checkpoint {checkpoint_file} (epoch {checkpoint['epoch']})")

    model = model.cuda()
    pred_res = test_epoch(model, 1, test_loader, tokenizer)
    with open(output_file, 'w') as f:
        for pred in pred_res:
            f.write(f"{pred}\n")


if __name__ == '__main__':
    infer20221212()
