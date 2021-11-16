import os
import random

import numpy as np
import torch
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import models
# from utils import datasets

default_workers = os.cpu_count()


def arch(config):
    if config['model_name'] in models.network_names:
        net = models.network_names[config['model_name']](**config['arch_kwargs'])
    else:
        raise ValueError(f'Invalid Arch: {config["arch"]}')

    return net


def nclass(config):
    r = {
        'imagenet100': 100,
        'cifar10': 10,
        'nuswide': 21,
        'coco': 80
    }[config['dataset']]

    return r


def optimizer(config, params):
    o_type = config['optima']
    kwargs = config['optima_kwargs']

    if o_type == 'sgd':
        o = SGD(params,
                lr=kwargs['lr'],
                momentum=kwargs.get('momentum', 0.9),
                weight_decay=kwargs.get('weight_decay', 0.0005),
                nesterov=kwargs.get('nesterov', False))
    else:  # adam
        o = Adam(params,
                 lr=kwargs['lr'],
                 betas=kwargs.get('betas', (0.9, 0.999)),
                 weight_decay=kwargs.get('weight_decay', 0))

    return o


def scheduler(config, optimizer):
    s_type = config['scheduler']
    kwargs = config['scheduler_kwargs']

    if s_type == 'step':
        return lr_scheduler.StepLR(optimizer,
                                   kwargs['step_size'],
                                   kwargs['gamma'])
    elif s_type == 'mstep':
        return lr_scheduler.MultiStepLR(optimizer,
                                        [int(float(m) * int(config['epochs'])) for m in
                                         kwargs['milestones'].split(',')],
                                        kwargs['gamma'])
    else:
        raise Exception('Scheduler not supported yet: ' + s_type)