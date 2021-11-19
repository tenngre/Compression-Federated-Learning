import os
import random

import numpy as np
import torch
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import models
from utils import datasets

default_workers = os.cpu_count()


def arch(config):
    if config['model_name'] in models.network_names:
        net = models.network_names[config['model_name']](**config['arch_kwargs'])
    else:
        raise ValueError(f'Invalid Arch: {config["model_name"]}')

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


def dataloader(d, bs=256, shuffle=True, workers=-1, drop_last=True):
    if workers < 0:
        workers = default_workers
    l = DataLoader(d,
                   bs,
                   shuffle,
                   drop_last=drop_last,
                   num_workers=workers)
    return l


def compose_transform(mode='train', resize=0, crop=0, norm=0,
                      augmentations=None):
    """
    :param mode:
    :param resize:
    :param crop:
    :param norm:
    :param augmentations:
    :return:
    if train:
      Resize (optional, usually done in Augmentations)
      Augmentations
      ToTensor
      Normalize
    if test:
      Resize
      CenterCrop
      ToTensor
      Normalize
    """
    # norm = 0, 0 to 1
    # norm = 1, -1 to 1
    # norm = 2, standardization
    mean, std = {
        0: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        1: [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        2: [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    }[norm]

    compose = []

    if resize != 0:
        compose.append(transforms.Resize(resize))

    if mode == 'train' and augmentations is not None:
        compose += augmentations

    if mode == 'test' and crop != 0 and resize != crop:
        compose.append(transforms.CenterCrop(crop))

    compose.append(transforms.ToTensor())

    if norm != 0:
        compose.append(transforms.Normalize(mean, std))

    return transforms.Compose(compose)


def dataset(config, filename, transform_mode):
    dataset_name = config['dataset']
    nclass = config['arch_kwargs']['nclass']

    resize = config['dataset_kwargs'].get('resize', 0)
    crop = config['dataset_kwargs'].get('crop', 0)
    norm = config['dataset_kwargs'].get('norm', 2)
    reset = config['dataset_kwargs'].get('reset', False)

    if dataset_name in ['imagenet100', 'nuswide', 'coco']:
        # resizec = 0 if resize == 256 else resize
        # cropc = 224 if crop == 0 else crop
        if transform_mode == 'train':
            transform = compose_transform('train', 0, crop, 2, {
                'imagenet100': [
                    transforms.RandomResizedCrop(crop),
                    # transforms.Resize(resize),
                    # transforms.RandomCrop(crop),
                    transforms.RandomHorizontalFlip()
                ],
                'nuswide': [
                    transforms.Resize(resize),
                    transforms.RandomCrop(crop),
                    transforms.RandomHorizontalFlip()
                ],
                'coco': [
                    transforms.Resize(resize),
                    transforms.RandomCrop(crop),
                    transforms.RandomHorizontalFlip()
                ]
            }[dataset_name])
        else:
            transform = compose_transform('test', resize, crop, 2)

        datafunc = {
            'imagenet100': datasets.imagenet100,
            'nuswide': datasets.nuswide,
            'coco': datasets.coco
        }[dataset_name]
        d = datafunc(transform=transform, filename=filename)

    else:  # cifar10/ cifar100
        resizec = 0 if resize == 32 else resize
        cropc = 0 if crop == 32 else crop

        if transform_mode == 'train':
            transform = compose_transform('train', resizec, 0, norm, [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
            ])
        else:
            transform = compose_transform('test', resizec, cropc, norm)
        ep = config['dataset_kwargs'].get('evaluation_protocol', 1)
        d = datasets.cifar(nclass, transform=transform, filename=filename, evaluation_protocol=ep, reset=reset)

    return d
