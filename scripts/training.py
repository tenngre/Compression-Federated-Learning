import copy
import json
import time
from collections import defaultdict
from datetime import datetime

import torch
import os
from .tools_noniid import ternary_convert, rec_w, zero_rates
import logging
import torch.nn as nn
import numpy as np
from utils.misc import AverageMeter, Timer
from pprint import pprint
from configs import *
import configs
from utils import io


class Client(object):
    def __init__(self, config, dataset=None, model=None, client_idx=None):
        self.loss_func = nn.CrossEntropyLoss()
        self.model = model
        self.local_train_dataset = dataloader(dataset, config['local_bs'])
        self.client_idx = client_idx
        self.optimizer = configs.optimizer(config, self.model.parameters())
        self.scheduler = configs.scheduler(config, self.optimizer)

    def train(self, config, r):
        net = self.model
        net.train()
        total_timer = Timer()
        timer = Timer()

        total_timer.tick()
        client_batch = []
        logging.info(f'Client {self.client_idx} is training on GPU {config["device"]}.')
        client_meters = defaultdict(AverageMeter)
        client_ep = {}

        if r != 0 and r % config['weights_decay_inter'] == 0:
            self.scheduler.step()

        for ep in range(config['local_ep']):
            meters = defaultdict(AverageMeter)
            res = {'ep': ep + 1}
            for batch_idx, (images, labels) in enumerate(self.local_train_dataset):
                timer.tick()

                images = images.to(config['device'])
                labels = labels.type(torch.LongTensor).to(config['device'])

                net.zero_grad()

                prob = net(images)
                loss = self.loss_func(prob, labels)
                loss.backward()
                self.optimizer.step()

                pre_labels = prob.data.max(1, keepdim=True)[1]

                # correct += pre_labels.eq(labels.data.view_as(pre_labels)).long().cpu().sum()
                correct = ((pre_labels.eq(labels.data.view_as(pre_labels)).long() * 1.0).cpu().mean()).item()
                # training_acc = correct.item() / len(self.local_train_dataset.dataset)

                timer.toc()

                # store results
                meters['loss_total'].update(loss.item(), images.size(0))
                meters['acc'].update(correct, images.size(0))
                meters['time'].update(timer.total)

                print(f'Local Epoch {ep} Client {self.client_idx} '
                      f'Train [{batch_idx + 1}/{len(self.local_train_dataset)}] '
                      f'Total Loss: {meters["loss_total"].avg:.4f} '
                      f'A(CE): {meters["acc"].avg:.2%} '
                      f'({timer.total:.2f}s / {total_timer.total:.2f}s)', end='\r')

            total_timer.toc()
            meters['total_time'].update(total_timer.total)

            for key in meters: res['train_' + key] = meters[key].avg
            client_batch.append(res)

        for item in client_batch:
            for key in item.keys():
                client_meters[f'Round_{r}_{self.client_idx}_{key}'].update(item[key])
        for key in client_meters:
            client_ep[key] = client_meters[key].avg

        if config['tnt_upload']:
            w_tnt, local_error = ternary_convert(copy.deepcopy(net))  # transmit tnt error
            return w_tnt, local_error, client_ep

        else:
            return net.state_dict(), client_ep


def test(model, config):
    model.eval()
    loss_func = nn.CrossEntropyLoss()

    meters = defaultdict(AverageMeter)
    total_timer = Timer()
    timer = Timer()
    total_timer.tick()

    # testing
    testing_loss = 0
    data_loader = configs.dataloader(config['test_set'], config['bs'], shuffle=False, drop_last=False)
    # data_loader = DataLoader(data_test, batch_size=config['bs'])

    logging.info(f'Testing on GPU {config["device"]}.')

    with torch.no_grad():
        for i, (data, labels) in enumerate(data_loader):
            timer.tick()
            data = data.to(config['device'])
            labels = labels.type(torch.LongTensor).to(config['device'])
            log_probs = model(data)

            # sum up batch loss
            testing_loss += loss_func(log_probs, labels).item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            # correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
            correct = ((y_pred.eq(labels.data.view_as(y_pred)).long() * 1.0).cpu().mean()).item()
            timer.toc()
            total_timer.toc()

            # store results
            meters['testing_loss_total'].update(testing_loss, data.size(0))
            meters['testing_acc'].update(correct, data.size(0))
            meters['time'].update(timer.total)

            print(f'Test [{i + 1}/{len(data_loader)}] '
                  f'T(loss): {meters["testing_loss_total"].avg:.4f} '
                  f'A(CE): {meters["testing_acc"].avg:.2%} '
                  f'({timer.total:.2f}s / {total_timer.total:.2f}s)', end='\r')
    meters['time'].update(total_timer.total)
    return meters


class Aggregator(object):
    def __init__(self, config):
        self.client_num = config['client_num']
        self.model = configs.arch(config).to(config['device'])
        self.model_name = config['model_name']
        self.zero_rate = False

    def inited_model(self):
        return self.model

    def client_model(self, model):
        cli_model = {}
        for idx in range(self.client_num):
            cli_model[str(idx)] = copy.deepcopy(model)
        return cli_model

    def params_aggregation(self, parma_dict):
        w = list(parma_dict.values())
        w_avg = copy.deepcopy(w[0])

        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], float(len(w)))
            # w_avg[k] = torch.nn.functional.normalize(torch.div(w_avg[k], float(len(w))), dim=0)

        return w_avg


def prepare_dataset(config):
    logging.info('Creating Datasets')
    train_dataset = configs.dataset(config, filename='train.txt', transform_mode='train')
    logging.info(f'Number of Train data: {len(train_dataset)}')
    test_dataset = configs.dataset(config, filename='test.txt', transform_mode='test')
    logging.info(f'Number of Test data: {len(test_dataset)}')

    return train_dataset, test_dataset


def clients_group(config, model):

    if os.path.exists('./client_index/client_index.txt'):
        if len(torch.load('./client_index/client_index.txt')) == config['client_num']:
            users_index = torch.load('./client_index/client_index.txt')
        else:
            print('[INFO] Existed client number not equal to input client number')
            time.sleep(2)
            print('[INFO] Generating NEW client index!!!')
            os.makedirs(f'./client_index', exist_ok=True)
            m = max(int(config['client_frac'] * config['client_num']), 1)
            users_index = np.random.choice(range(config['client_num']), m, replace=False)
            torch.save(users_index, f'./client_index/client_index.txt')
    else:
        print('[INFO] Client index DO NOT EXIST!!')
        print('[INFO] Generating client index')
        os.makedirs(f'./client_index', exist_ok=True)
        m = max(int(config['client_frac'] * config['client_num']), 1)
        users_index = np.random.choice(range(config['client_num']), m, replace=False)
        torch.save(users_index, f'./client_index/client_index.txt')

    train_dataset, test_dataset = prepare_dataset(config)

    config['train_set'] = train_dataset
    config['test_set'] = test_dataset

    client_group = {}
    for idx in users_index:
        client = Client(config=config,
                        dataset=train_dataset[idx],
                        model=copy.deepcopy(model),
                        client_idx=idx)
        client_group[idx] = client

    return client_group


def average(lst):
    return sum(lst) / len(lst)


def main_tnt_upload(config):
    logdir = config['logdir']
    assert logdir != '', 'please input logdir'

    pprint(config)

    os.makedirs(f'{logdir}/models', exist_ok=True)
    os.makedirs(f'{logdir}/optims', exist_ok=True)
    os.makedirs(f'{logdir}/outputs', exist_ok=True)
    json.dump(config, open(f'{logdir}/config.json', 'w+'), indent=4, sort_keys=True)

    aggregator = Aggregator(config)
    print('Building model..')
    inited_model = aggregator.inited_model()
    print(inited_model)

    print('Init Clients')
    client_group = clients_group(config, inited_model)

    round_train = {}
    round_test = {}
    round_time = []

    train_history = []
    test_history = []

    compression_rate = {}

    glob_norm = {}

    nepochs = config['epochs']
    neval = config['eval_interval']

    best = 0
    curr_metric = 0

    for epoch in range(config['epochs']):
        start_time = time.time()
        client_upload = {}
        client_local = {}
        train_acc_total = []
        train_loss_total = []
        # local_zero_rates = []

        print(f'\n | Global Training Round: {epoch}|\n')

        # training
        for idx in client_group.keys():
            ter_params, qua_error, client_ep = client_group[idx].train(config, epoch)
            client_local[idx] = copy.deepcopy(qua_error)
            client_upload[idx] = copy.deepcopy(ter_params)
            z_r = zero_rates(ter_params)
            # local_zero_rates.append(z_r)
            # logging.info(f'Client {idx} zero rate {z_r:.2%}')
            compression_rate[f'Round_{epoch}_Client_{idx}_compression_rate'] = z_r

            # recording local training info
            train_history.append(client_ep)

            # recording local training info
            train_acc_total.append(client_ep[f'Round_{epoch}_{idx}_train_acc'])
            train_loss_total.append(client_ep[f'Round_{epoch}_{idx}_train_loss_total'])

        round_train[f'{epoch}_Round_train_acc'] = average(train_acc_total)
        round_train[f'{epoch}_Round_train_lose'] = average(train_loss_total)

        json.dump(train_history, open(f'{logdir}/train_history.json', 'w+'), indent=True, sort_keys=True)
        json.dump(round_train, open(f'{logdir}/train_round_history.json', 'w+'), indent=True, sort_keys=True)
        json.dump(compression_rate, open(f'{logdir}/client_compression_rate.json', 'w+'), indent=True, sort_keys=True)

        # aggregation in server
        glob_avg = aggregator.params_aggregation(copy.deepcopy(client_upload))

        glob_norm[f'Round_{epoch}'] = {}
        for key in glob_avg.keys():
            norm_ = torch.norm(glob_avg[key])
            glob_norm[f'Round_{epoch}'][key] = norm_.item()

        json.dump(glob_norm, open(f'{logdir}/glob_norm.json', 'w+'), indent=True, sort_keys=True)

        # update local models
        for idx in client_group.keys():
            client_group[idx].model = rec_w(copy.deepcopy(glob_avg),
                                            copy.deepcopy(client_local[idx]),
                                            client_group[idx].model)

        # client testing
        eval_now = (epoch + 1) == nepochs or (neval != 0 and (epoch + 1) % neval == 0)
        if eval_now:
            client_round_acc = []
            client_round_loss = []
            test_history_local = {'Round': epoch + 1}
            logging.info(f'\n |Round {epoch} Client Test |\n')
            for idx in client_group.keys():
                logging.info(f'Client {idx} Testing on GPU {config["device"]}.')
                testing_res = test(model=client_group[idx].model,
                                   config=config)
                test_history_local[f'client_{idx}'] = {}
                test_history_local[f'client_{idx}']['test_acc'] = testing_res['testing_acc'].avg
                test_history_local[f'client_{idx}']['test_lose'] = testing_res['testing_loss_total'].avg

                client_round_acc.append(testing_res['testing_acc'].avg)
                client_round_loss.append(testing_res['testing_loss_total'].avg)
            test_history.append(test_history_local)

            curr_metric = average(client_round_acc)
            round_test[f'{epoch}_Round_test_acc'] = average(client_round_acc)
            round_test[f'{epoch}_Round_test_lose'] = average(client_round_loss)

            if len(test_history) != 0:
                json.dump(test_history, open(f'{logdir}/test_history.json', 'w+'), indent=True, sort_keys=True)
                json.dump(round_test, open(f'{logdir}/test_round_history.json', 'w+'), indent=True, sort_keys=True)

            elapsed = time.time() - start_time
            round_time.append(elapsed)

            print(f'Round {epoch} costs time: {elapsed:.2f}s|'
                  f'Train Acc.: {round_train[f"{epoch}_Round_train_acc"]:.2%}| '
                  f'Test Acc.{round_test[f"{epoch}_Round_test_acc"]:.2%}| '
                  f'Train loss: {round_train[f"{epoch}_Round_train_lose"]:.4f}| '
                  f'Test loss: {round_test[f"{epoch}_Round_test_lose"]:.4f}| ')

        modelsd = inited_model.state_dict()
        # optimsd = optimizer.state_dict()
        # io.fast_save(modelsd, f'{logdir}/models/last.pth')
        # io.fast_save(optimsd, f'{logdir}/optims/last.pth')
        save_now = config['save_interval'] != 0 and (epoch + 1) % config['save_interval'] == 0
        if save_now:
            torch.save(modelsd, f'{logdir}/models/ep{epoch + 1}.pth')
            # io.fast_save(optimsd, f'{logdir}/optims/ep{ep + 1}.pth')
            # io.fast_save(train_outputs, f'{logdir}/outputs/train_ep{ep + 1}.pth')

        if best < curr_metric:
            best = curr_metric
            torch.save(modelsd, f'{logdir}/models/best.pth')
            # io.fast_save(modelsd, f'{logdir}/models/best.pth')

    modelsd = inited_model.state_dict()
    torch.save(modelsd, f'{logdir}/models/last.pth')
    total_time = time.time() - start_time
    # io.join_save_queue()
    logging.info(f'Training End at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}')
    logging.info(f'Total time used: {total_time / (60 * 60):.2f} hours')
    logging.info(f'Best mAP: {best:.6f}')
    logging.info(f'Done: {logdir}')


def main_norm_upload(config):
    logdir = config['logdir']
    assert logdir != '', 'please input logdir'

    pprint(config)

    os.makedirs(f'{logdir}/models', exist_ok=True)
    os.makedirs(f'{logdir}/optims', exist_ok=True)
    os.makedirs(f'{logdir}/outputs', exist_ok=True)
    json.dump(config, open(f'{logdir}/config.json', 'w+'), indent=4, sort_keys=True)

    aggregator = Aggregator(config)
    print('Building model..')
    inited_model = aggregator.inited_model()
    print(inited_model)

    print('Init Clients')
    client_group = clients_group(config, inited_model)

    round_train = {}
    round_test = {}
    round_time = []
    train_history = []
    test_history = []

    compression_rate = {}

    nepochs = config['epochs']
    neval = config['eval_interval']

    best = 0
    curr_metric = 0

    for epoch in range(config['epochs']):
        start_time = time.time()
        client_upload = {}
        train_acc_total = []
        train_loss_total = []

        print(f'\n | Global Training Round: {epoch} Training|\n')

        # training
        for idx in client_group.keys():
            w_, client_ep = client_group[idx].train(config, epoch)
            client_upload[idx] = copy.deepcopy(w_)
            z_r = zero_rates(w_)
            compression_rate[f'Round_{epoch}_Client_{idx}_compression_rate'] = z_r

            train_history.append(client_ep)

            # recording local training info
            train_acc_total.append(client_ep[f'Round_{epoch}_{idx}_train_acc'])
            train_loss_total.append(client_ep[f'Round_{epoch}_{idx}_train_loss_total'])

        round_train[f'{epoch}_Round_train_acc'] = average(train_acc_total)
        round_train[f'{epoch}_Round_train_loss'] = average(train_loss_total)

        json.dump(train_history, open(f'{logdir}/train_history.json', 'w+'), indent=True, sort_keys=True)
        json.dump(round_train, open(f'{logdir}/train_round_history.json', 'w+'), indent=True, sort_keys=True)
        json.dump(compression_rate, open(f'{logdir}/client_compression_rate.json', 'w+'), indent=True, sort_keys=True)

        # aggregation in server
        glob_avg = aggregator.params_aggregation(copy.deepcopy(client_upload))
        inited_model.load_state_dict(glob_avg)

        # update local models
        for idx in client_group.keys():
            client_group[idx].model.load_state_dict(glob_avg)

        # ====model testing===

        # eval_now = (epoch + 1) == nepochs or (neval != 0 and (ep + 1) % neval == 0)
        # if eval_now:
        print(f'\n |Round {epoch} Global Test|\n')
        testing_res = test(model=inited_model,
                           config=config)

        temp_res = {}
        for key in testing_res.keys():
            temp_res[key] = testing_res[key].avg
        test_history.append(temp_res)

        curr_metric = testing_res['testing_acc'].avg

        round_test[f'{epoch}_Round_test_acc'] = testing_res['testing_acc'].avg
        round_test[f'{epoch}_Round_test_lose'] = testing_res['testing_loss_total'].avg

        if len(test_history) != 0:
            json.dump(test_history, open(f'{logdir}/test_history.json', 'w+'), indent=True, sort_keys=True)
            json.dump(round_test, open(f'{logdir}/test_round_history.json', 'w+'), indent=True, sort_keys=True)

        elapsed = time.time() - start_time
        round_time.append(elapsed)

        print(f'Round {epoch} costs time: {elapsed:.2f}s|'
              f'Train Acc.: {round_train[f"{epoch}_Round_train_acc"]:.2%}| '
              f'Test Acc.{round_test[f"{epoch}_Round_test_acc"]:.2%}| '
              f'Train loss: {round_train[f"{epoch}_Round_train_loss"]:.4f}| '
              f'Test loss: {round_test[f"{epoch}_Round_test_lose"]:.4f}| ')

        modelsd = inited_model.state_dict()
        # optimsd = optimizer.state_dict()
        # io.fast_save(modelsd, f'{logdir}/models/last.pth')
        # io.fast_save(optimsd, f'{logdir}/optims/last.pth')
        save_now = config['save_interval'] != 0 and (epoch + 1) % config['save_interval'] == 0
        if save_now:
            torch.save(modelsd, f'{logdir}/models/ep{epoch + 1}.pth')
            # io.fast_save(optimsd, f'{logdir}/optims/ep{ep + 1}.pth')
            # io.fast_save(train_outputs, f'{logdir}/outputs/train_ep{ep + 1}.pth')

        if best < curr_metric:
            best = curr_metric
            torch.save(modelsd, f'{logdir}/models/best.pth')
            # io.fast_save(modelsd, f'{logdir}/models/best.pth')

    modelsd = inited_model.state_dict()
    torch.save(modelsd, f'{logdir}/models/last.pth')
    total_time = time.time() - start_time
    # io.join_save_queue()
    logging.info(f'Training End at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}')
    logging.info(f'Total time used: {total_time / (60 * 60):.2f} hours')
    logging.info(f'Best mAP: {best:.6f}')
    logging.info(f'Done: {logdir}')
