import copy
import json
import time

import torch
from torch.utils.data import DataLoader
import os

from scripts import DatasetSplit, FedAvg, zero_rates, ternary_convert, current_learning_rate
from utils.utils import progress_bar
import torch.nn as nn
import numpy as np

train_acc, train_loss = [], []
test_acc, test_loss = [], []
train_time = []
comp_rate = []
update_zero_rate = []


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, client=None, device=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.client = client
        self.device = device

    def train(self, net, lr):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=self.args.lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        #         optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)

        epoch_loss = []
        epoch_acc = []
        total = 0
        print('Client {} is training on GPU {}.'.format(self.client, self.device))
        for i in range(self.args.local_ep):
            batch_loss = []
            batch_acc = 0
            correct = 0
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # print(images[0])
                #                 if self.args.tnt_image:
                #                     images = TNT.image_tnt(images)
                images = images
                labels = labels.type(torch.LongTensor)
                net.zero_grad()
                log_probs = net(images)
                # print(type(log_probs))
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                total += labels.size(0)

                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
                train_acc = correct.item() / len(self.ldr_train.dataset)
                batch_acc = train_acc
                batch_loss.append(loss.item())

                progress_bar(batch_idx, len(self.ldr_train), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                             % (sum(batch_loss) / (batch_idx + 1), train_acc * 100., correct.item(),
                                len(self.ldr_train.dataset)))

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(batch_acc)

        return net, sum(epoch_loss) / len(epoch_loss), epoch_acc[-1]


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


def main(config, args, client_net):
    for epoch in range(args.epochs):
        start_time = time.time()
        client_upload = {}
        client_local = {}
        acc_locals_train = {}
        loss_locals_train = []
        acc_locals_test = {}
        local_zero_rates = []

        print(f'c\n | Global Training Round: {epoch} Training {args.his}|\n')
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # training
        for idx in idxs_users:
            local = LocalUpdate(args=args,
                                dataset=config['trainset'],
                                idxs=np.int_(config['client_traindata'][idx]),
                                client=idx)
            network, loss_local_train, acc_local_train = local.train(net=client_net[str(idx)].to(config['device']),
                                                                     lr=config['current_lr'])
            # Global TNT weights or Norm Weights
            if args.tntupload:
                print('ternary update')
                w_tnt, local_error = ternary_convert(copy.deepcopy(client_net[str(idx)]))  # transmit tnt error
                client_local[str(idx)] = copy.deepcopy(local_error)
                client_upload[str(idx)] = copy.deepcopy(w_tnt)
                z_r = zero_rates(w_tnt)
                local_zero_rates.append(z_r)
                print('Client {} zero rate {:.2%}'.format(idx, z_r))
            else:
                client_upload[str(idx)] = copy.deepcopy(client_net[str(idx)].state_dict())

            # recording local training info
            acc_locals_train[str(idx)] = copy.deepcopy(acc_local_train)
            loss_locals_train.append(copy.deepcopy(loss_local_train))
        elapsed = time.time() - start_time
        train_time.append(elapsed)

        # aggregation in server
        glob_avg, cr = FedAvg(copy.deepcopy(client_upload), copy.deepcopy(acc_locals_train), 1)

        print('Global Zero Rates {:.2%}'.format(cr))
        comp_rate.append(cr)

        # update local models
        if args.tntupload:
            for idx in idxs_users:
                client_net[str(idx)] = rec_w(copy.deepcopy(glob_avg),
                                             copy.deepcopy(client_local[str(idx)]),
                                             client_net[str(idx)])
        else:
            for idx in idxs_users:
                client_net[str(idx)].load_state_dict(glob_avg)

        # local testing
        if args.tntupload:
            print(f'\n |Round {epoch} Client Test {args.his}|\n')
            client_acc = []
            client_loss = []
            for idx in idxs_users:
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
        else:
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
