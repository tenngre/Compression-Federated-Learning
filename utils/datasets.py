import os

import numpy as np
import torch
import math
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets.folder import pil_loader

DATA_FOLDER = {
    # 'nuswide': 'data/nuswide_v2_256_resize',  # resize to 256x256
    # 'imagenet': 'data/imagenet_resize',  # resize to 224x224
    'cifar': './data/cifar'  # auto generate
    # 'coco': 'data/coco',
    # 'gldv2': 'data/gldv2delgembed',
    # 'roxf': 'data/roxford5kdelgembed',
    # 'rpar': 'data/rparis5kdelgembed'
}


def cifar_iid(nclass, **kwargs):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """

    transform = kwargs['transform']
    ep = kwargs['evaluation_protocol']
    fn = kwargs['filename']
    reset = kwargs['reset']
    num_users = kwargs['num_users']

    prefix = DATA_FOLDER['cifar']

    CIFAR = CIFAR10 if int(nclass) == 10 else CIFAR100
    traind = CIFAR(f'{prefix}{nclass}',
                   transform=transform,
                   train=True,
                   download=True)
    testd = CIFAR(f'{prefix}{nclass}', train=False, download=True)

    combine_data = np.concatenate([traind.data, testd.data], axis=0)
    combine_targets = np.concatenate([traind.targets, testd.targets], axis=0)

    path = f'{prefix}{nclass}/iid_{ep}_{fn}'

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

            query_n = 1000  # // (nclass // 10)

            index_for_query = index_of_class[:query_n].tolist()
            index_for_db = index_of_class[query_n:].tolist()
            index_for_train = index_for_db

            train_data_index.extend(index_for_train)
            query_data_index.extend(index_for_query)
            db_data_index.extend(index_for_db)

        # train_data_index = np.array(train_data_index)
        query_data_index = np.array(query_data_index)
        db_data_index = np.array(db_data_index)

        num_items = int(len(train_data_index) / num_users)  # number of items in one client
        dict_client_data_index = {}  # dict_user for recording the client number;

        for i in range(num_users):  # choosing training data for each client
            dict_client_data_index[i] = set(np.random.choice(train_data_index,
                                                             num_items,
                                                             replace=False))  # randomly choose ##num_items## items
            # for a client
            # without replacing
            train_data_index = list(set(train_data_index) - dict_client_data_index[i]) # remove selected items
            dict_client_data_index[i] = np.array(list(dict_client_data_index[i]))

        torch.save(dict_client_data_index, f'./data/cifar{nclass}/iid_{ep}_train.txt')
        torch.save(query_data_index, f'./data/cifar{nclass}/iid_{ep}_test.txt')
        torch.save(db_data_index, f'./data/cifar{nclass}/iid_{ep}_database.txt')

        client_data = {}
        if fn == 'train.txt':
            for idx in range(num_users):
                traind.data = combine_data[dict_client_data_index[idx]]
                traind.targets = combine_targets[dict_client_data_index[idx]]
                client_data[idx] = traind
            return client_data
        else:
            data_index = {
                'test.txt': query_data_index,
                'database.txt': db_data_index
            }[fn]

            traind.data = combine_data[data_index]
            traind.targets = combine_targets[data_index]

            return traind  # contains the training data for each client i


def cifar(nclass, **kwargs):
    transform = kwargs['transform']
    ep = kwargs['evaluation_protocol']
    fn = kwargs['filename']
    reset = kwargs['reset']

    prefix = DATA_FOLDER['cifar']

    CIFAR = CIFAR10 if int(nclass) == 10 else CIFAR100
    traind = CIFAR(f'{prefix}{nclass}',
                   transform=transform,
                   train=True,
                   download=True)
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

        torch.save(train_data_index, f'./data/cifar{nclass}/0_{ep}_train.txt')
        torch.save(query_data_index, f'./data/cifar{nclass}/0_{ep}_test.txt')
        torch.save(db_data_index, f'./data/cifar{nclass}/0_{ep}_database.txt')

        data_index = {
            'train.txt': train_data_index,
            'test.txt': query_data_index,
            'database.txt': db_data_index
        }[fn]

    traind.data = combine_data[data_index]
    traind.targets = combine_targets[data_index]

    return traind


def cifar_non_iid(train_dataset, test_dataset, config):
    num_users = config['client_num']
    n_class = config['n_class']

    num_shards_train = num_users * n_class  # minimum shard needed
    num_classes = 10
    num_imgs_perc_test, num_imgs_test_total = 1000, 10000
    assert (n_class * num_users <= num_shards_train)
    assert (n_class <= num_classes)
    idx_class = [i for i in range(num_classes)]

    # Generate the minimum whole dataset that is needed for training
    # eg: class 5 client 5, 25 shards, minimum whole dataset = 3
    idx_shard = [i % 10 for i in range(math.ceil(
        num_shards_train / 10) * 10)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(50000)  # 5000 sample per class
    # labels = dataset.train_labels.numpy()
    labels = np.array(train_dataset.targets)
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)
    # labels_test_raw = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    labels = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    # print(idxs_labels_test[0, :])
    # print(idxs_labels_test[1, :])

    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        # randomly pick non-repetitive classes from the dataset
        rand_set = set(np.random.choice(np.unique(idx_shard), n_class, replace=False))
        # print(idx_shard)
        # print(rand_set)
        for rand in rand_set:
            # Remove the first occurance of the chosen class
            idx_shard.remove(rand)

            # Get all samples from each class that has been chosen
            dict_users_train[i] = np.concatenate(
                (dict_users_train[i], idxs[rand * 5000:(rand + 1) * 5000]), axis=0)
            user_labels = np.concatenate((user_labels, labels[rand * 5000:(rand + 1) * 5000]),
                                         axis=0)
        print((dict_users_train[i][0]))
        user_labels_set = set(user_labels)
        # print(user_labels_set)
        # print(user_labels)
        for label in user_labels_set:
            # print(label)
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs_test[int(label) * num_imgs_perc_test:int(label + 1) * num_imgs_perc_test]),
                axis=0)
        # print(set(labels_test_raw[dict_users_test[i].astype(int)]))

    return dict_users_train, dict_users_test
