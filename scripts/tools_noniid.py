from ternay.convert_tnt import *
import copy
import math
import numpy as np
from torch.utils.data import Dataset


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)  # numbers of items in one client
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]  # dict_user for recording the client number;
    # all_idxs is the indx of each item
    for i in range(num_users):  # choosing training data for each client
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))  # randomly choose ##num_items## items for a client
        # without replacing
        all_idxs = list(set(all_idxs) - dict_users[i])  # remove selected items
    return dict_users  # contains the training data for each client i


def cifar_extr_noniid(train_dataset, test_dataset, num_users, n_class):
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


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, np.int_(label)


def current_learning_rate(epoch, current_lr, config):
    if (epoch + 1) % config['d_epoch'] == 0:
        return current_lr * 0.1
    else:
        return current_lr


def ternary_convert(network):
    net_error = copy.deepcopy(network)  # create a network for saving local errors
    w = copy.deepcopy(network.state_dict())  # for local weights
    w_error = copy.deepcopy(net_error.state_dict())  # for local errors

    for name, module in network.named_modules():  # calculating errors between normal weights and errors
        if isinstance(module, TNTConv2d):
            #             print('convolution ternary')
            w[name + str('.weight')] = KernelsCluster.apply(module.weight)  # tnt
            w_error[name + str('.weight')] -= w[name + str('.weight')]  # errors

        if isinstance(module, TNTLinear):
            #             print('linear ternary')
            w[name + str('.weight')] = KernelsCluster.apply(module.weight)
            w_error[name + str('.weight')] -= w[name + str('.weight')]

    network.load_state_dict(w, strict=False)  # load tnt to model
    tnt_weights_dict = network.state_dict()  # creating a tensor form dict

    net_error.load_state_dict(w_error, strict=False)
    tnt_error_dict = net_error.state_dict()
    t = zero_rates(tnt_weights_dict)
    print('[INFO] zero rates is ', t)

    return tnt_weights_dict, tnt_error_dict


def float_pass(tnt, w, network):
    pass_w = copy.deepcopy(w)
    for name, module in network.named_modules():
        if isinstance(module, TNTConv2d):
            mask = (tnt[name + str('.weight')] == 0).float()
            pass_w[name + str('.weight')] = w[name + str('.weight')] * mask
    return pass_w


def FedAvg(w_dict, acc_dict, num):
    #     for i in range(num):
    #         lowest_acc_idx = min(acc_dict, key=acc_dict.get)
    #         w_dict.pop(lowest_acc_idx)
    #         acc_dict.pop(lowest_acc_idx)
    w = list(w_dict.values())

    w_avg = copy.deepcopy(w[0])
    # for k in w_avg.keys():
    #     w_count = (w_avg[k] != 0).float()
    #     for i in range(1, len(w)):
    #         w_avg[k] += w[i][k]
    #         mask = (w[i][k] != 0).float()
    #         w_count += mask
    #     w_avg[k] = torch.div(w_avg[k], w_count + 1e-7)

    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], float(len(w)))

    total_params = 0
    zero_params = 0
    for key in w_avg.keys():
        zero_params += (w_avg[key].view(-1) == 0).sum().item()
        total_params += len(w_avg[key].view(-1))

    return w_avg, (zero_params / total_params)


def rec_w(avg_tnt, local_err, net):
    # recover weights from avg_tnt and local_err
    for name, module in net.named_modules():
        if isinstance(module, TNTConv2d):
            avg_tnt[name + str('.weight')] += local_err[name + str('.weight')]
        if isinstance(module, TNTLinear):
            avg_tnt[name + str('.weight')] += local_err[name + str('.weight')]
    net.load_state_dict(avg_tnt)
    return net


def zero_rates(weight_dict):
    total_params = 0
    zero_params = 0
    for key in weight_dict.keys():
        #         print(key)
        zero_params += (weight_dict[key].view(-1) == 0).sum().item()
        total_params += len(weight_dict[key].view(-1))
    return zero_params / total_params


def store_weights(model):
    model_dict = {}
    for name, param in model.named_parameters():
        model_dict[name] = param.data.clone()
        # param.requires_grad = False
    return model_dict


def WeightsUpdate(global_new, global_old, omiga):
    old_weights = store_weights(global_old)
    new_weights = store_weights(global_new)

    for key in old_weights.keys():
        old_weights[key] += omiga * new_weights[key]

    for name, param in global_old.named_parameters():
        param.data.copy_(old_weights[name])
        # param.requires_grad = True
    # return global_old



