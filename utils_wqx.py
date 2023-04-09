import torch
import os
import json
import numpy as np
from rich import print



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(state, outname, local_rank):
    if local_rank == 0:
        # best_acc = state['best_acc']
        # epoch = state['epoch']
        # filename = 'checkpoint_acc_%.4f_epoch_%02d.pth.tar' % (best_acc, epoch)
        filename = outname
        # filename = 'checkpoint_best_%d.pth.tar'
        # filename = os.path.join('output/', filename)
        dir_name = os.path.dirname(filename)
        os.makedirs(dir_name, exist_ok=True)
        torch.save(state, filename)

        # best_filename = os.path.join(model_dir, 'checkpoint_best_%d.pth.tar' % name_no)
        # best_filename = filename
        # shutil.copyfile(filename, best_filename)
        print('=> Save model to %s' % filename)
