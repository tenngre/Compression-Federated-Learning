import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets.folder import pil_loader

DATA_FOLDER = {
    # 'nuswide': 'data/nuswide_v2_256_resize',  # resize to 256x256
    # 'imagenet': 'data/imagenet_resize',  # resize to 224x224
    'cifar': 'data/cifar'  # auto generate
    # 'coco': 'data/coco',
    # 'gldv2': 'data/gldv2delgembed',
    # 'roxf': 'data/roxford5kdelgembed',
    # 'rpar': 'data/rparis5kdelgembed'
}


def cifar(nclass, **kwargs):
    transform = kwargs['transform']
    ep = kwargs['evaluation_protocol']
    fn = kwargs['filename']
    reset = kwargs['reset']

    prefix = DATA_FOLDER['cifar']

    CIFAR = CIFAR10 if int(nclass) == 10 else CIFAR100
    traind = CIFAR(f'{prefix}{nclass}',
                   transform=transform, target_transform=one_hot(int(nclass)),
                   train=True, download=True)
    testd = CIFAR(f'{prefix}{nclass}', train=False, download=True)

    combine_data = np.concatenate([traind.data, testd.data], axis=0)
    combine_targets = np.concatenate([traind.targets, testd.targets], axis=0)

    path = f'{prefix}{nclass}/0_{ep}_{fn}'

    load_data = fn == 'train.txt'
    load_data = load_data and (reset or not os.path.exists(path))

    if not load_data:
        print(f'Loading {path}')
        data_index = torch.load(path)
    else:
        train_data_index = []
        query_data_index = []
        db_data_index = []

        data_id = np.arange(combine_data.shape[0])  # [0, 1, ...]

        for i in range(nclass):
            class_mask = combine_targets == i
            index_of_class = data_id[class_mask].copy()  # index of the class [2, 10, 656,...]
            np.random.shuffle(index_of_class)

            if ep == 1:
                query_n = 100  # // (nclass // 10)
                train_n = 500  # // (nclass // 10)

                index_for_query = index_of_class[:query_n].tolist()

                index_for_db = index_of_class[query_n:].tolist()
                index_for_train = index_for_db[:train_n]
            else:  # ep2 = take all data
                query_n = 1000  # // (nclass // 10)

                index_for_query = index_of_class[:query_n].tolist()
                index_for_db = index_of_class[query_n:].tolist()
                index_for_train = index_for_db

            train_data_index.extend(index_for_train)
            query_data_index.extend(index_for_query)
            db_data_index.extend(index_for_db)

        train_data_index = np.array(train_data_index)
        query_data_index = np.array(query_data_index)
        db_data_index = np.array(db_data_index)

        torch.save(train_data_index, f'data/cifar{nclass}/0_{ep}_train.txt')
        torch.save(query_data_index, f'data/cifar{nclass}/0_{ep}_test.txt')
        torch.save(db_data_index, f'data/cifar{nclass}/0_{ep}_database.txt')

        data_index = {
            'train.txt': train_data_index,
            'test.txt': query_data_index,
            'database.txt': db_data_index
        }[fn]

    traind.data = combine_data[data_index]
    traind.targets = combine_targets[data_index]

    return traind
