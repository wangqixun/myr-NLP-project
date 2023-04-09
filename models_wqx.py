import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch
import os
from collections import OrderedDict
import copy


class MyModel(nn.Module):
    def __init__(self, nlp_model, mode):
        super().__init__()
        self.nlp_model = nlp_model
        if mode == 'large':
            nb_feature = 1024
        else:
            nb_feature = 768

        self.cls = nn.Linear(nb_feature, 2)
    
    def forward(self, **data):
        x = self.nlp_model(**data)
        y = self.cls(x['last_hidden_state'][:, 0])
        return y


def get_model(cfg, mode='large'):
    # tokenizer = AutoTokenizer.from_pretrained(cfg['common_encoder_pretrained_transformers'], additional_special_tokens=added_token)
    # tokenizer = BertTokenizer.from_pretrained(cfg['common_encoder_pretrained_transformers'], use_fast=True, cache_dir='/share/wangqixun/workspace/bs/tx_mm/code/cache', additional_special_tokens=added_token)
    tokenizer = AutoTokenizer.from_pretrained(cfg['pretrained_transformers'], use_fast=True, cache_dir=cfg['cache_dir'])

    nlp_model = AutoModel.from_pretrained(cfg['pretrained_transformers'], cache_dir=cfg['cache_dir'])
    model = MyModel(nlp_model=nlp_model, mode=mode)

    return_dict = {
        'model': model,
        'tokenizer': tokenizer,
    }
    return return_dict

