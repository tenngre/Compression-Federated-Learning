import copy
import json
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import os

from scripts import DatasetSplit, zero_rates, ternary_convert, current_learning_rate, rec_w
from tnt_fl_train_noniid import dataset_test
from utils.utils import progress_bar
import torch.nn as nn
import numpy as np
from utils.misc import AverageMeter, Timer


import configs

train_acc, train_loss = [], []
test_acc, test_loss = [], []
train_time = []
comp_rate = []
update_zero_rate = []


class Client(object):
    def __init__(self, args, dataset=None, idxs=None, client=None, device=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.client = client
        self.device = device
        self.ternary_convert = args.tnt_upload

    def train(self, config, net):
        net.train()
        optimizer = configs.optimizer(config, net.parameters())

        meters = defaultdict(AverageMeter)
        total_timer = Timer()
        timer = Timer()

        total_timer.tick()

        epoch_loss = []
        epoch_acc = []
        total = 0
        print(f'Client {self.client} is training on GPU {self.device}.')

        for i in range(self.args.local_ep):
            batch_loss = []
            batch_acc = 0
            correct = 0
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                timer.tick()

                images = images
                labels = labels.type(torch.LongTensor)

                net.zero_grad()

                prob = net(images)
                loss = self.loss_func(prob, labels)
                loss.backward()
                optimizer.step()

                pred_labels = prob.data.max(1, keepdim=True)[1]

                total += labels.size(0)
                correct += pred_labels.eq(labels.data.view_as(pred_labels)).long().cpu().sum()
                train_acc = correct.item() / len(self.ldr_train.dataset)

                batch_acc = train_acc
                batch_loss.append(loss.item())

                timer.toc()
                total_timer.toc()

                # store results
                meters['loss_total'].update(loss.item(), images.size(0))
                meters['acc'].update(train_acc, images.size(0))
                meters['time'].update(timer.total)

                print(f'Train [{batch_idx + 1}/{len(self.ldr_train)}] '
                      f'Total Loss: {meters["loss_total"].avg:.4f} '
                      f'A(CE): {meters["acc"].avg:.2%} '
                      f'({timer.total:.2f}s / {total_timer.total:.2f}s)', end='\r')

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(batch_acc)

        if self.args.tnt_upload:
            w_tnt, local_error = ternary_convert(copy.deepcopy(net))  # transmit tnt error
            return w_tnt, local_error, sum(epoch_loss) / len(epoch_loss), epoch_acc[-1]

        else:

            return net.state_dict(), sum(epoch_loss) / len(epoch_loss), epoch_acc[-1]


def test_img(idxs, epoch, net_g, datatest, args, best_acc, dict_users_test=None):
    net_g.eval()

    # testing
    test_loss = 0
    correct = 0
    total = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)

    print('Client {} Testing on GPU {}.'.format(idxs, args.GPU))
    for idx, (data, target) in enumerate(data_loader):
        #         if args.tnt_image:
        #             data = TNT.image_tnt(data)
        data = data.to(torch.device("cuda:" + str(args.GPU)))
        target = target.to(torch.device("cuda:" + str(args.GPU)))
        log_probs = net_g(data)

        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        total += target.size(0)

        progress_bar(idx, len(data_loader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                     % (test_loss / total, 100. * correct / total, correct, total))

    test_loss /= len(data_loader.dataset)
    acc = correct.item() / len(data_loader.dataset)

    # saving best
    if acc > best_acc:
        print('Saving..')
        state = {
            # 'net': net_g.get_tnt(),  # net_g.get_tnt(),  # 'net':net.get_tnt() for tnt network // net.state_dict()
            'net': net_g.get_tnt() if args.tntupload else net_g.state_dict(),
            # net_g.get_tnt(),  # 'net':net.get_tnt() for tnt network // net.state_dict()
            'acc': acc * 100.,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}.ckpt'.format(args.his))
        best_acc = acc

    if args.save:
        dict_name = args.his.split('.')[0]
        path = os.path.join('./saved/', '{}/epoch_{}_{}.ckpt'.format(dict_name, epoch, args.his))
        if epoch % 10 == 0:
            print('Saving..')
            state = {
                # 'net': net_g.get_tnt(),
                'net': net_g.get_tnt() if args.tntupload else net_g.state_dict(),
                # net_g.get_tnt(),  # 'net':net.get_tnt() for tnt network // net.state_dict()
                'acc': acc * 100.,
                'epoch': epoch,
            }
            if not os.path.isdir('./saved/{}'.format(dict_name)):
                os.makedirs('./saved/{}'.format(dict_name))
            torch.save(state, path)
            best_acc = acc
    return acc, test_loss, best_acc


class Aggregator(object):
    def __init__(self, config, args):
        self.client_num = config['client_num']
        self.model = config['Model'][args.model](10).to(config['device'])
        self.zero_rate = False

    def inited_model(self):
        return self.model

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


def clients_group(config, args):
    m = max(int(args.frac * args.num_users), 1)
    users_index = np.random.choice(range(args.num_users), m, replace=False)

    client_group = {}
    for idx in users_index:
        client = Client(args=args,
                        dataset=config['trainset'],
                        idxs=np.int_(config['client_traindata'][idx]),
                        client=idx)
        client_group[idx] = client

    return client_group


def main_tnt_upload(config, args):
    aggregator = Aggregator(config, args)
    print('==> Building model..')
    inited_mode = aggregator.inited_model()
    print(inited_mode)

    print('Init Clients')
    client_group = clients_group(config, args)

    print('Deliver model to clients')
    client_net = aggregator.client_model(inited_mode)

    for epoch in range(args.epochs):
        start_time = time.time()
        client_upload = {}
        client_local = {}
        acc_locals_train = {}
        loss_locals_train = []
        acc_locals_test = {}
        local_zero_rates = []

        print(f'\n | Global Training Round: {epoch} Training {args.his}|\n')

        # training
        for idx in client_group.keys():
            w_tnt, local_error, loss_local_train, acc_local_train = client_group[idx].train(config,
                                                                                            net=client_net[
                                                                                                str(idx)].to(
                                                                                                config['device']))
            client_local[str(idx)] = copy.deepcopy(local_error)
            client_upload[str(idx)] = copy.deepcopy(w_tnt)
            z_r = zero_rates(w_tnt)
            local_zero_rates.append(z_r)
            print('Client {} zero rate {:.2%}'.format(idx, z_r))

            # recording local training info
            acc_locals_train[str(idx)] = copy.deepcopy(acc_local_train)
            loss_locals_train.append(copy.deepcopy(loss_local_train))

            # recording local training info
            acc_locals_train[str(idx)] = copy.deepcopy(acc_local_train)
            loss_locals_train.append(copy.deepcopy(loss_local_train))
        elapsed = time.time() - start_time
        train_time.append(elapsed)

        # aggregation in server
        glob_avg = aggregator.params_aggregate(copy.deepcopy(client_upload))

        # print('Global Zero Rates {:.2%}'.format(cr))
        # comp_rate.append(cr)

        # update local models
        for idx in client_group.keys():
            client_net[str(idx)] = rec_w(copy.deepcopy(glob_avg),
                                         copy.deepcopy(client_local[str(idx)]),
                                         client_net[str(idx)])

        # local testing
        print(f'\n |Round {epoch} Client Test {args.his}|\n')
        client_acc = []
        client_loss = []
        for idx in client_group.keys():
            acc_t, loss_t, best_acc = test_img(idx,
                                               epoch,
                                               client_net[str(idx)],
                                               dataset_test,
                                               args,
                                               best_acc)
            client_acc.append(acc_t)
            client_loss.append(loss_t)
        test_acc.append(sum(client_acc) / len(idxs_users))
        test_loss.append(sum(client_loss) / len(idxs_users))

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

        #     writer.add_scalar("Loss/train", loss, epoch)
        #     writer.flush()
        print('Round {} costs time: {:.2f}s| Train Acc.: {:.2%}| '
              'Test Acc.{:.2%}| Train loss: {:.4f}| Test loss: {:.4f}| '
              'Down Rate is {:.3%}| Up Rate{:.3%}'
              ' Floating agg {}'.format(epoch,
                                        elapsed,
                                        avg_acc_train,
                                        test_acc[-1],
                                        loss_avg,
                                        test_loss[-1],
                                        cr,
                                        temp_zero_rates,
                                        config['glob_agg_num']))

        current_lr = current_learning_rate(epoch, current_lr, args)

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
    with open('./his/{}.json'.format(args.his), 'w+') as f:
        json.dump(his_dict, f, indent=2)


def main_norm_upload(config, args):
    aggregator = Aggregator(config, args)
    print('==> Building model..')
    inited_mode = aggregator.inited_model()
    print(inited_mode)

    print('Init Clients')
    client_group = clients_group(config, args)

    print('Deliver model to clients')
    client_net = aggregator.client_model(inited_mode)

    for epoch in range(args.epochs):
        start_time = time.time()
        client_upload = {}
        client_local = {}
        acc_locals_train = {}
        loss_locals_train = []
        acc_locals_test = {}
        local_zero_rates = []

        print(f'\n | Global Training Round: {epoch} Training {args.his}|\n')

        # training
        for idx in client_group.keys():
            w_, loss_local_train, acc_local_train = client_group[idx].train(config,
                                                                            net=client_net[str(idx)].to(
                                                                                config['device']))
            client_upload[str(idx)] = copy.deepcopy(w_)

            # recording local training info
            acc_locals_train[str(idx)] = copy.deepcopy(acc_local_train)
            loss_locals_train.append(copy.deepcopy(loss_local_train))

            # recording local training info
            acc_locals_train[str(idx)] = copy.deepcopy(acc_local_train)
            loss_locals_train.append(copy.deepcopy(loss_local_train))

        elapsed = time.time() - start_time
        train_time.append(elapsed)

        # aggregation in server
        glob_avg = aggregator.params_aggregate(copy.deepcopy(client_upload))

        # print('Global Zero Rates {:.2%}'.format(cr))
        # comp_rate.append(cr)

        # update local models

        for idx in client_group.keys():
            client_net[str(idx)].load_state_dict(glob_avg)

        # local testing
        print(f'\n |Round {epoch} Global Test {args.his}|\n')
        acc_t, loss_t, best_acc = test_img('all', epoch, client_net['0'],
                                           dataset_test,
                                           args,
                                           best_acc)
        test_acc.append(acc_t)
        test_loss.append(loss_t)

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

        #     writer.add_scalar("Loss/train", loss, epoch)
        #     writer.flush()
        print('Round {} costs time: {:.2f}s| Train Acc.: {:.2%}| '
              'Test Acc.{:.2%}| Train loss: {:.4f}| Test loss: {:.4f}| '
              'Down Rate is {:.3%}| Up Rate{:.3%}'
              ' Floating agg {}'.format(epoch,
                                        elapsed,
                                        avg_acc_train,
                                        test_acc[-1],
                                        loss_avg,
                                        test_loss[-1],
                                        cr,
                                        temp_zero_rates,
                                        config['glob_agg_num']))

        current_lr = current_learning_rate(epoch, current_lr, args)

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
    with open('./his/{}.json'.format(args.his), 'w+') as f:
        json.dump(his_dict, f, indent=2)
