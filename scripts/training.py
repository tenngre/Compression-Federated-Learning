import copy
import json
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import os

from scripts import DatasetSplit, zero_rates, ternary_convert, current_learning_rate, rec_w
from tnt_fl_train_noniid import dataset_test
import logging
import torch.nn as nn
import numpy as np

from utils.datasets import cifar_non_iid
from utils.misc import AverageMeter, Timer
import torch.nn.functional as F
from pprint import pprint

import configs

train_acc, train_loss = [], []
test_acc, test_loss = [], []
train_time = []
comp_rate = []
update_zero_rate = []


class Client(object):
    def __init__(self, config, dataset=None, idxs=None, client=None):
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs),
                                    batch_size=config['local_bs'],
                                    shuffle=True)
        self.client = client
        self.device = config['device']
        self.ternary_convert = config['tnt_upload']

    def train(self, config, net):
        net.train()
        optimizer = configs.optimizer(config, net.parameters())

        total_timer = Timer()
        timer = Timer()

        total_timer.tick()
        epoch_res = []

        print(f'Client {self.client} is training on GPU {self.device}.')
        for ep in range(config['local_ep']):
            meters = defaultdict(AverageMeter)
            res = {'ep': ep + 1}
            correct = 0
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                timer.tick()

                images = images.to(config['device'])
                labels = labels.type(torch.LongTensor).to(config['device'])

                net.zero_grad()

                prob = net(images)
                loss = self.loss_func(prob, labels)
                loss.backward()
                optimizer.step()

                pre_labels = prob.data.max(1, keepdim=True)[1]

                correct += pre_labels.eq(labels.data.view_as(pre_labels)).long().cpu().sum()
                training_acc = correct.item() / len(self.ldr_train.dataset)

                timer.toc()
                total_timer.toc()

                # store results
                meters['loss_total'].update(loss.item(), images.size(0))
                meters['acc'].update(training_acc, images.size(0))
                meters['time'].update(timer.total)

                print(f'Train [{batch_idx + 1}/{len(self.ldr_train)}] '
                      f'Total Loss: {meters["loss_total"].avg:.4f} '
                      f'A(CE): {meters["acc"].avg:.2%} '
                      f'({timer.total:.2f}s / {total_timer.total:.2f}s)', end='\r')

            for key in meters: res['train_' + key] = meters[key].avg

            epoch_res.append(res)

        if config['tnt_upload']:
            w_tnt, local_error = ternary_convert(copy.deepcopy(net))  # transmit tnt error
            return w_tnt, local_error, epoch_res

        else:

            return net.state_dict(), epoch_res


def test(model, data_test, config):
    model.eval()

    meters = defaultdict(AverageMeter)
    total_timer = Timer()
    timer = Timer()
    total_timer.tick()

    # testing
    testing_loss = 0
    correct = 0
    total = 0
    data_loader = DataLoader(data_test, batch_size=config['bs'])

    print('Testing on GPU {}.'.format(config['device']))
    for i, (data, labels) in enumerate(data_loader):
        timer.tick()

        with torch.no_grad():
            data, labels = data.to(config['device']), labels.to(config['device'])
            log_probs = model(data)

            # sum up batch loss
            testing_loss += F.cross_entropy(log_probs, labels, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
            total += labels.size(0)
        timer.toc()
        total_timer.toc()

        acc = correct.item() / len(data_loader.dataset)

        # store results
        meters['testing_loss_total'].update(testing_loss, data.size(0))
        meters['testing_acc'].update(acc, data.size(0))
        meters['time'].update(timer.total)

        print(f'Test [{i + 1}/{len(data_loader)}] '
              f'T(loss): {meters["testing_loss_total"].avg:.4f} '
              f'A(CE): {meters["testing_acc"].avg:.2%} '
              f'({timer.total:.2f}s / {total_timer.total:.2f}s)', end='\r')

    return meters

    # # saving best
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         # 'net': net_g.get_tnt(),  # net_g.get_tnt(),  # 'net':net.get_tnt() for tnt network // net.state_dict()
    #         'net': net_g.get_tnt() if config['tnt_upload'] else net_g.state_dict(),
    #         # net_g.get_tnt(),  # 'net':net.get_tnt() for tnt network // net.state_dict()
    #         'acc': acc * 100.,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/{}.ckpt'.format(config['history']))
    #     best_acc = acc
    #
    # if config['save']:
    #     dict_name = config['history'].split('.')[0]
    #     path = os.path.join('./saved/', '{}/epoch_{}_{}.ckpt'.format(dict_name, epoch, config['history']))
    #     if epoch % 10 == 0:
    #         print('Saving..')
    #         state = {
    #             # 'net': net_g.get_tnt(),
    #             'net': net_g.get_tnt() if config['tnt_upload'] else net_g.state_dict(),
    #             # net_g.get_tnt(),  # 'net':net.get_tnt() for tnt network // net.state_dict()
    #             'acc': acc * 100.,
    #             'epoch': epoch,
    #         }
    #         if not os.path.isdir('./saved/{}'.format(dict_name)):
    #             os.makedirs('./saved/{}'.format(dict_name))
    #         torch.save(state, path)
    #         best_acc = acc
    # return acc, test_loss, best_acc


class Aggregator(object):
    def __init__(self, config):
        self.client_num = config['client_num']
        self.model = configs.arch(config).to(config['device'])
        self.model_name = config['model_name']
        self.zero_rate = False

    def inited_model(self):
        model = self.model
        return model

    def client_model(self, model):
        cli_model = {}
        for idx in range(self.client_num):
            cli_model[str(idx)] = copy.deepcopy(model)
        return cli_model

    def params_aggregate(self, parma_dict):
        w = list(parma_dict.values())
        w_avg = copy.deepcopy(w[0])

        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], float(len(w)))

        if self.zero_rate:
            total_params = 0
            zero_params = 0
            for key in w_avg.keys():
                zero_params += (w_avg[key].view(-1) == 0).sum().item()
                total_params += len(w_avg[key].view(-1))

            return w_avg, (zero_params / total_params)

        else:
            return w_avg


def clients_group(config):
    m = max(int(config['client_frac'] * config['client_num']), 1)
    users_index = np.random.choice(range(config['client_num']), m, replace=False)
    print(config['client_train_data'])

    train_dataset = prepare_dataloader(config)
    test_dataset = prepare_dataloader(config)
    dict_users_train, dict_users_test = cifar_non_iid(train_dataset,
                                                      test_dataset,
                                                      config)
    config['client_train_data'] = config['client_train_data'].get('client_train_data', train_dataset)
    config['client_test_data'] = config['client_test_data'].get('client_test_data', dict_users_test)

    client_group = {}
    for idx in users_index:
        client = Client(config=config,
                        dataset=config['train_set'],
                        idxs=np.int_(config['client_train_data'][idx]),
                        client=idx)
        client_group[idx] = client

    return client_group


def prepare_dataloader(config):
    logging.info('Creating Datasets')
    train_dataset = configs.dataset(config, filename='train.txt', transform_mode='train')
    logging.info(f'Number of Train data: {len(train_dataset)}')

    test_dataset = configs.dataset(config, filename='test.txt', transform_mode='test')

    train_loader = configs.dataloader(train_dataset, config['batch_size'])
    test_loader = configs.dataloader(test_dataset, config['batch_size'], shuffle=False, drop_last=False)

    return train_loader, test_loader


def main_tnt_upload(config):
    logdir = config['logdir']
    assert logdir != '', 'please input logdir'

    pprint(config)

    os.makedirs(f'{logdir}/models', exist_ok=True)
    os.makedirs(f'{logdir}/optims', exist_ok=True)
    os.makedirs(f'{logdir}/outputs', exist_ok=True)
    json.dump(config, open(f'{logdir}/config.json', 'w+'), indent=4, sort_keys=True)

    aggregator = Aggregator(config)

    print('==> Building model..')
    inited_mode = aggregator.inited_model()
    print(inited_mode)
    print('Deliver model to clients')
    client_net = aggregator.client_model(inited_mode)

    print('Init Clients')
    client_group = clients_group(config)
    current_lr = config['current_lr']

    for epoch in range(config['epochs']):
        start_time = time.time()
        client_upload = {}
        client_local = {}
        acc_locals_train = {}
        loss_locals_train = []
        local_zero_rates = []

        print(f'\n | Global Training Round: {epoch} Training {config["history"]}|\n')

        # training
        for idx in client_group.keys():
            ter_params, qua_error, res = client_group[idx].train(config,
                                                                 net=client_net[str(idx)].to(config['device']))
            client_local[str(idx)] = copy.deepcopy(qua_error)
            client_upload[str(idx)] = copy.deepcopy(ter_params)
            z_r = zero_rates(ter_params)
            local_zero_rates.append(z_r)
            print('Client {} zero rate {:.2%}'.format(idx, z_r))

            # recording local training info
            acc_locals_train[str(idx)] = copy.deepcopy(res[-1]['train_acc'])
            loss_locals_train.append(copy.deepcopy(res[-1]['train_loss_total']))

        elapsed = time.time() - start_time
        train_time.append(elapsed)

        # aggregation in server
        glob_avg = aggregator.params_aggregate(copy.deepcopy(client_upload))

        # update local models
        for idx in client_group.keys():
            client_net[str(idx)] = rec_w(copy.deepcopy(glob_avg),
                                         copy.deepcopy(client_local[str(idx)]),
                                         client_net[str(idx)])

        # client testing
        print(f'\n |Round {epoch} Client Test {config["history"]}|\n')
        client_acc = []
        client_loss = []
        for idx in client_group.keys():
            print('Client {} Testing on GPU {}.'.format(idx, config['device']))
            testing_res = test(model=client_net[str(idx)],
                               data_test=dataset_test,
                               config=config)

            test_acc.append(testing_res['testing_acc'])
            test_loss.append(testing_res['testing_loss_total'])

            client_acc.append(testing_res['testing_acc'].avg)
            client_loss.append(testing_res['testing_loss_total'].avg)
        test_acc.append(sum(client_acc) / len(client_group))
        test_loss.append(sum(client_loss) / len(client_group))

        # training info update
        avg_acc_train = sum(acc_locals_train.values()) / len(acc_locals_train.values())

        train_acc.append(avg_acc_train)

        loss_avg = sum(loss_locals_train) / len(loss_locals_train)
        train_loss.append(loss_avg)
        try:
            temp_zero_rates = sum(local_zero_rates) / len(local_zero_rates)

        except:
            temp_zero_rates = sum(local_zero_rates)
        update_zero_rate.append(temp_zero_rates)

        print(f'Round {epoch} costs time: {elapsed:.2f}s| Train Acc.: {avg_acc_train:.2%}| '
              f'Test Acc.{test_acc[-1]:.2%}| Train loss: {loss_avg:.4f}| Test loss: {test_loss[-1]:.4f}| '
              f'| Up Rate{temp_zero_rates:.3%}')

        current_lr = current_learning_rate(epoch, current_lr, config)

    his_dict = {
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'test_loss': test_loss,
        'test_correct': test_acc,
        'train_time': train_time,
        'glob_zero_rates': comp_rate,
        'local_zero_rates': update_zero_rate,

    }

    os.makedirs('./his/', exist_ok=True)
    with open('./his/{}.json'.format(config["history"]), 'w+') as f:
        json.dump(his_dict, f, indent=2)


def main_norm_upload(config):
    aggregator = Aggregator(config)
    print('==> Building model..')
    inited_mode = aggregator.inited_model()
    print(inited_mode)

    print('Init Clients')
    client_group = clients_group(config)

    print('Deliver model to clients')
    client_net = aggregator.client_model(inited_mode)

    best_acc = 0
    current_lr = config['current_lr']

    for epoch in range(config['epochs']):
        start_time = time.time()
        client_upload = {}
        acc_locals_train = {}
        loss_locals_train = []
        local_zero_rates = []

        print(f'\n | Global Training Round: {epoch} Training {config["history"]}|\n')

        # training
        for idx in client_group.keys():
            w_, res = client_group[idx].train(config,
                                              net=client_net[str(idx)].to(config['device']))
            client_upload[str(idx)] = copy.deepcopy(w_)

            # recording local training info
            acc_locals_train[str(idx)] = copy.deepcopy(res[-1]['train_acc'])
            loss_locals_train.append(copy.deepcopy(res[-1]['train_loss_total']))

        elapsed = time.time() - start_time
        train_time.append(elapsed)

        # aggregation in server
        glob_avg = aggregator.params_aggregate(copy.deepcopy(client_upload))

        # update local models

        for idx in client_group.keys():
            client_net[str(idx)].load_state_dict(glob_avg)

        # ====model testing===

        print(f'\n |Round {epoch} Global Test {config["history"]}|\n')
        testing_res = test(model=client_net[str(0)],
                           data_test=dataset_test,
                           config=config)

        test_acc.append(testing_res['testing_acc'])
        test_loss.append(testing_res['testing_loss_total'])

        # training info update
        avg_acc_train = sum(acc_locals_train.values()) / len(acc_locals_train.values())

        train_acc.append(avg_acc_train)

        loss_avg = sum(loss_locals_train) / len(loss_locals_train)
        train_loss.append(loss_avg)
        try:
            temp_zero_rates = sum(local_zero_rates) / len(local_zero_rates)

        except:
            temp_zero_rates = sum(local_zero_rates)
        update_zero_rate.append(temp_zero_rates)

        print(f'Round {epoch} costs time: {elapsed:.2f}s| Train Acc.: {avg_acc_train:.2%}| '
              f'Test Acc.{test_acc[-1].avg:.2%}| Train loss: {loss_avg:.4f}| Test loss: {test_loss[-1].avg:.4f}| '
              f'| Up Rate{temp_zero_rates:.3%}')

        # current_lr = current_learning_rate(epoch, current_lr, config)

    his_dict = {
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'test_loss': test_loss,
        'test_correct': test_acc,
        'train_time': train_time,
        'glob_zero_rates': comp_rate,
        'local_zero_rates': update_zero_rate,
    }

    os.makedirs('./his/', exist_ok=True)
    with open('./his/{}.json'.format(config["history"]), 'w+') as f:
        json.dump(his_dict, f, indent=2)
