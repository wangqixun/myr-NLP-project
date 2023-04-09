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
# from collections import OrderedDict
from rich import print
import time
import cv2
# from glob import glob
import string
from transformers import AdamW, get_linear_schedule_with_warmup

# from dataset_wqx import ChusaiDataset
# from utils_wqx import save_checkpoint, write_json, load_json
# from models_wqx import get_model
# from category_id_map import category_id_to_lv2id
# from category_id_map import lv2id_to_lv1id
# from losses import multilabel_categorical_crossentropy, crossentropy, sparse_categorical_crossentropy_with_prior, ce_and_cce, iou_loss, celv12

from models_wqx import get_model
from dataset_wqx import MyDataset
from utils_wqx import save_checkpoint, AverageMeter, ProgressMeter

if __name__ == '__main__':
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    scaler = GradScaler()
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


print_raw = print
def print(*info):
    if local_rank == 0:
        print_raw(*info)




def crossentropy(y_true, y_pred):
    return F.cross_entropy(y_pred, y_true, label_smoothing=0.2)


def evaluate(predictions, labels):
    nb_all = len(predictions)
    acc = sum([int(p==l) for p, l in zip(predictions, labels)]) / (nb_all + 1e-8)

    eval_results = {'acc': acc}
    return eval_results


def train_epoch(model, optimizer, epoch, dataloader, sampler, tokenizer, scheduler):
    print(f"\n\n=> train")
    data_time = AverageMeter('- data', ':4.3f')
    batch_time = AverageMeter('- batch', ':6.3f')
    losses = AverageMeter('- loss', ':.4e')
    acces = AverageMeter('- acc', ':.4f')
    progress = ProgressMeter(
        len(dataloader), data_time, batch_time, losses, acces, prefix=f"Epoch: [{epoch}]")

    end = time.time()
    model.train()
    sampler.set_epoch(epoch)

    predictions, labels = [], []

    for batch_index, data_batch in enumerate(dataloader):
        optimizer.zero_grad()

        context_str_batch, target_batch = data_batch

        # data tokenizer
        context_token_batch = tokenizer(context_str_batch, padding=True, truncation=True, max_length=500, return_tensors='pt')
        
        # to gpu
        context_token_batch = {k:v.to(device) for k,v in context_token_batch.items()}
        target_batch = target_batch.to(device)

        # forward
        data_input_batch = context_token_batch
        output_batch = model(**data_input_batch)

        pred_batch = output_batch.softmax(dim=-1)

        loss_batch = crossentropy(target_batch, output_batch)
        loss = torch.mean(loss_batch)
        # print(loss)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_value = loss.item()
        losses.update(loss_value, len(target_batch))
        pred = torch.argmax(pred_batch, dim=-1)
        predictions.extend(pred.cpu().numpy())
        labels.extend(target_batch.cpu().numpy())
        acc_batch = (target_batch==pred).sum().cpu().numpy() / (len(target_batch) + 1e-8)
        acces.update(acc_batch, len(target_batch))
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_index % 50 == 0:
            progress.print(batch_index)

    results = evaluate(predictions, labels)
    print(results)
    return results


def val_epoch(model, optimizer, epoch, dataloader, sampler, tokenizer):
    print(f"\n\n=> val")
    data_time = AverageMeter('- data', ':4.3f')
    batch_time = AverageMeter('- batch', ':6.3f')
    losses = AverageMeter('- loss', ':.4e')
    acces = AverageMeter('- acc', ':.4f')
    progress = ProgressMeter(
        len(dataloader), data_time, batch_time, losses, acces, prefix=f"Epoch: [{epoch}]")

    end = time.time()
    model.train()
    sampler.set_epoch(epoch)

    predictions, labels = [], []

    for batch_index, data_batch in enumerate(dataloader):
        optimizer.zero_grad()

        context_str_batch, target_batch = data_batch

        # data tokenizer
        context_token_batch = tokenizer(context_str_batch, padding=True, truncation=True, max_length=500, return_tensors='pt')
        
        # to gpu
        context_token_batch = {k:v.to(device) for k,v in context_token_batch.items()}
        target_batch = target_batch.to(device)

        # forward
        data_input_batch = context_token_batch
        output_batch = model(**data_input_batch)

        pred_batch = output_batch.softmax(dim=-1)

        loss_batch = crossentropy(target_batch, output_batch)
        loss = torch.mean(loss_batch)
        # print(pred_batch)
        # print(target_batch)
        # print(loss)

        loss_value = loss.item()
        losses.update(loss_value, len(target_batch))
        pred = torch.argmax(pred_batch, dim=-1)
        predictions.extend(pred.cpu().numpy())
        labels.extend(target_batch.cpu().numpy())
        acc_batch = (target_batch==pred).sum().cpu().numpy() / (len(target_batch) + 1e-8)
        acces.update(acc_batch, len(target_batch))
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_index % 50 == 0:
            progress.print(batch_index)

    results = evaluate(predictions, labels)
    print(results)
    return results



def gogogo():
    # /share/wangqixun/workspace/bs/tx_mm/code/model_dl/junnyu/roformer_v2_chinese_char_base

    # 数据参数
    # pretrained_transformers = '/share/wangqixun/workspace/bs/tx_mm/code/model_dl/hfl/chinese-roberta-wwm-ext'
    pretrained_transformers = '/share/wangqixun/workspace/github_project/transformers_checkpoint/hfl/chinese-macbert-base'
    # pretrained_transformers = '/share/wangqixun/workspace/github_project/transformers_checkpoint/hfl/chinese-pert-base'
    # pretrained_transformers = '/share/wangqixun/workspace/github_project/transformers_checkpoint/hfl/chinese-lert-base'
    # pretrained_transformers = '/share/wangqixun/workspace/github_project/transformers_checkpoint/hfl/chinese-macbert-large'
    output_dir = '/share/wangqixun/workspace/bs/myr/output/v3_macbert'
    
    # 训练、验证数据
    ann_file_tra = '/share/wangqixun/workspace/bs/myr/data/tra.csv'
    ann_file_val = '/share/wangqixun/workspace/bs/myr/data/val.csv'

    checkpoint_file = None

    # 策略参数
    batch_size = 4
    epochs = 15
    cache_dir = '/share/wangqixun/workspace/bs/myr/output/tmp'

    model_cfg = dict(
        pretrained_transformers=pretrained_transformers,
        cache_dir=cache_dir,
    )
    # 模型
    model_dict = get_model(model_cfg, mode='large')
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    print(model)

    # 优化器
    # optimizer = torch.optim.AdamW(model.parameters(), 1e-5)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-6)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=int(20999/2/batch_size))
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    scheduler = None

    # dataloader
    data_loader_cfg = {}
    tra_dataset = MyDataset(ann_file_tra, data_loader_cfg, mode='tra')
    data_val_loader_cfg = {}
    val_dataset = MyDataset(ann_file_val, data_val_loader_cfg, mode='val')
    sampler_tra = DistributedSampler(tra_dataset, shuffle=True)
    sampler_val = DistributedSampler(val_dataset, shuffle=False)
    tra_loader = DataLoader(tra_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, sampler=sampler_tra)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, sampler=sampler_val)

    # resume
    if checkpoint_file is not None and os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        init_epoch = checkpoint['epoch'] + 1
        # model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        optimizer.load_state_dict(checkpoint['optimizer'])
        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        print(f"=> Resume: loaded checkpoint {checkpoint_file} (epoch {checkpoint['epoch']})")
    else:
        init_epoch = 1
        print(f"=> No checkpoint. ")

    # model to gpu
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 训练
    acc = 0.
    for epoch in range(init_epoch, epochs + 1):
        results_tra = train_epoch(model, optimizer, epoch, tra_loader, sampler_tra, tokenizer, scheduler)
        results_val  = val_epoch(model, optimizer, epoch, val_loader, sampler_tra, tokenizer)
        acc_val = results_val['acc']
        if acc_val >= acc:
            acc = acc_val
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc': acc,
                'optimizer': optimizer.state_dict(),
            }, outname=f'{output_dir}/checkpoint_epoch{epoch:03d}_acc{acc:.4f}.pth.tar', local_rank=local_rank)


if __name__ == '__main__':
    gogogo()
