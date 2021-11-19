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
    with torch.no_grad():
        qua_net = copy.deepcopy(network)  # create a network for quantify errors
        qua_params = copy.deepcopy(network.state_dict())  # for quantify parameters
        qua_error = copy.deepcopy(qua_net.state_dict())  # for local errors

        for name, module in network.named_modules():  # calculating errors between normal weights and errors
            if isinstance(module, TNTConv2d):
                qua_params[name + str('.weight')] = KernelsCluster.apply(module.weight)  # tnt
                qua_error[name + str('.weight')] -= qua_params[name + str('.weight')]  # errors

            if isinstance(module, TNTLinear):
                qua_params[name + str('.weight')] = KernelsCluster.apply(module.weight)
                qua_error[name + str('.weight')] -= qua_params[name + str('.weight')]

        network.load_state_dict(qua_params, strict=False)  # load tnt to model
        qua_net.load_state_dict(qua_error, strict=False)  # load quantify error

        ter_weights_dict = network.state_dict()  # creating a tensor form dict
        qua_error_dict = qua_net.state_dict()

    return ter_weights_dict, qua_error_dict


def float_pass(tnt, w, network):
    pass_w = copy.deepcopy(w)
    for name, module in network.named_modules():
        if isinstance(module, TNTConv2d):
            mask = (tnt[name + str('.weight')] == 0).float()
            pass_w[name + str('.weight')] = w[name + str('.weight')] * mask
    return pass_w


def rec_w(global_avg, qua_err, network):
    # recover weights from avg_tnt and local_err
    for name, module in network.named_modules():
        if isinstance(module, TNTConv2d):
            global_avg[name + str('.weight')] += qua_err[name + str('.weight')]
        if isinstance(module, TNTLinear):
            global_avg[name + str('.weight')] += qua_err[name + str('.weight')]
    network.load_state_dict(global_avg)
    return network


def zero_rates(weight_dict):
    total_params = 0
    zero_params = 0
    for key in weight_dict.keys():
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
