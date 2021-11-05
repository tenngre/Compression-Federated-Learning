import torchvision
import torchvision.transforms as transforms
from models import *
from scripts.tools_noniid import *
import json
import os
import argparse
import random
import numpy as np
from scripts import training


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--his', type=str, required=True)
parser.add_argument('--num_users', default=10, type=int, )
parser.add_argument('--epochs', default=100, help='epoch', type=int)
parser.add_argument('--frac', default=1, type=int)
parser.add_argument('--local_bs', default=128, type=int)
parser.add_argument('--save', action='store_true', help='save model every 10 epoch')
parser.add_argument('--GPU', default=0, type=int)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--split', default='user')
parser.add_argument('--local_ep', default=1, type=int)
parser.add_argument('--bs', default=128, type=int)
parser.add_argument('--d_epoch', default=50, type=int)
parser.add_argument('--decay_r', default=0.1, type=float)
parser.add_argument('--tntupload', action='store_true', help='uploading tnt weights')
parser.add_argument('--weight_decay', default=0.0001, type=float)
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default='res18_norm', type=str)
parser.add_argument('--n_class', default=2, type=int, help='class number in each client')
# parser.add_argument('--num_samples', default=200, type=int)
parser.add_argument('--g_c', default=200, type=int, help='floating model communication epoch')
args = parser.parse_args()

args_dict = vars(args)
os.makedirs('./setting', exist_ok=True)
with open('./setting/config_{}.json'.format(args.his), 'w+') as f:
    json.dump(args_dict, f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# torch.cuda.set_device(device)


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset_train = torchvision.datasets.CIFAR10(root='/data/datasets/cifar10',
                                             train=True, download=True,
                                             transform=transform_train)

dataset_test = torchvision.datasets.CIFAR10(root='/data/datasets/cifar10',
                                            train=False, download=True,
                                            transform=transform_test)


def random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


random_seed(80)

# train_dataset, test_dataset, num_users, n_class, num_samples, rate_unbalance
dict_users_train, dict_users_test = cifar_extr_noniid(dataset_train, dataset_test,
                                                      args.num_users, args.n_class)

# Model
print('==> Building model..')
Model = {
    'vgg_tnt': VGG_tnt,
    'vgg_norm': VGG_norm,
    'mobilev2_tnt': MobileNetV2_tnt,
    'mobilev2_norm': MobileNetV2,
    'res18_tnt': ResNet_TNT18,
    'res18_norm': ResNet18,
    'res50_tnt': ResNet_TNT50,
    'res50_norm': ResNet50,
    'alex_tnt': AlexNet_tnt,
    'alex_norm': AlexNet
}

net_glob = Model[args.model](10).to(device)
print(net_glob)

# training
config = {
    'current_lr': args.lr,
    'acc_rate': 0.5,
    'best_acc': 0,
    'glob_agg_num': 0,
    'trainset': dataset_train,
    'testset': dataset_test,
    'client_traindata': dict_users_train,
    'client_testdata': dict_users_test,
    'device': device,

    'client_num': 10,

    'optima': 'adam',
    'optima_kwargs': {
        'lr': args.lr,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'nesterov': False,
        'betas': (0.9, 0.999)
    },
    'scheduler': 'step',
    'scheduler_kwargs': {
        'step_size': int(args.epochs * 0.8),
        'gamma': 0.1,
        'milestones': '0.5,0.75'
    },
    'Model': {
        'vgg_tnt': VGG_tnt,
        'vgg_norm': VGG_norm,
        'mobilev2_tnt': MobileNetV2_tnt,
        'mobilev2_norm': MobileNetV2,
        'res18_tnt': ResNet_TNT18,
        'res18_norm': ResNet18,
        'res50_tnt': ResNet_TNT50,
        'res50_norm': ResNet50,
        'alex_tnt': AlexNet_tnt,
        'alex_norm': AlexNet}
}


client_net = Client_net(net_glob, args.num_users)


# def acc_rate(train_acc):
#     return abs((train_acc[-1] - train_acc[-2]) / (train_acc[-2] - train_acc[-3]))


if __name__ == '__main__':
    training.main(config, args, client_net)

# for epoch in range(args.epochs):
#     start_time = time.time()
#     client_upload = {}
#     client_local = {}
#     acc_locals_train = {}
#     loss_locals_train = []
#     acc_locals_test = {}
#     local_zero_rates = []
#
#     print(f'c\n | Global Training Round: {epoch} Training {args.his}|\n')
#     m = max(int(args.frac * args.num_users), 1)
#     idxs_users = np.random.choice(range(args.num_users), m, replace=False)
#
#     # training
#     for idx in idxs_users:
#         local = LocalUpdate(args=args, dataset=dataset_train, idxs=np.int_(dict_users_train[idx]), client=idx)
#         network, loss_local_train, acc_local_train = local.train(net=client_net[str(idx)].to(device), lr=current_lr)
#         # Global TNT weights or Norm Weights
#         if args.tntupload:
#             # if (epoch+1) % args.g_c == 0:
#             #     print('floating update')
#             #     client_upload[str(idx)] = copy.deepcopy(client_net[str(idx)].state_dict())
#             #     local_zero_rates.append(0)
#             # else:
#             print('ternary update')
#             w_tnt, local_error = ternary_convert(copy.deepcopy(client_net[str(idx)]))  # transmit tnt error
#             client_local[str(idx)] = copy.deepcopy(local_error)
#             client_upload[str(idx)] = copy.deepcopy(w_tnt)
#             z_r = zero_rates(w_tnt)
#             local_zero_rates.append(z_r)
#             print('Client {} zero rate {:.2%}'.format(idx, z_r))
#         else:
#             client_upload[str(idx)] = copy.deepcopy(client_net[str(idx)].state_dict())
#
#         # recording local training info
#         acc_locals_train[str(idx)] = copy.deepcopy(acc_local_train)
#         loss_locals_train.append(copy.deepcopy(loss_local_train))
#     elapsed = time.time() - start_time
#     train_time.append(elapsed)
#
#     # aggregation in server
#     glob_avg, cr = FedAvg(copy.deepcopy(client_upload), copy.deepcopy(acc_locals_train), 1)
#
#     print('Global Zero Rates {:.2%}'.format(cr))
#     comp_rate.append(cr)
#
#     # update local models
#     if args.tntupload:
#         for idx in idxs_users:
#             client_net[str(idx)] = rec_w(copy.deepcopy(glob_avg),
#                                          copy.deepcopy(client_local[str(idx)]),
#                                          client_net[str(idx)])
#     else:
#         for idx in idxs_users:
#             client_net[str(idx)].load_state_dict(glob_avg)
#
#     # local testing
#     if args.tntupload:
#         print(f'\n |Round {epoch} Client Test {args.his}|\n')
#         client_acc = []
#         client_loss = []
#         for idx in idxs_users:
#             acc_t, loss_t, best_acc = test_img(idx, epoch, client_net[str(idx)],
#                                                dataset_test, args, best_acc)
#             client_acc.append(acc_t)
#             client_loss.append(loss_t)
#         test_acc.append(sum(client_acc) / len(idxs_users))
#         test_loss.append(sum(client_loss) / len(idxs_users))
#     else:
#         print(f'\n |Round {epoch} Global Test {args.his}|\n')
#         acc_t, loss_t, best_acc = test_img('all', epoch, client_net['0'], dataset_test, args, best_acc)
#         test_acc.append(acc_t)
#         test_loss.append(loss_t)
#
#     # training info update
#     avg_acc_train = sum(acc_locals_train.values()) / len(acc_locals_train.values())
#     #     print(train_acc)
#     train_acc.append(avg_acc_train)
#     #     try:
#     #         print('[INFO] acc. rate', abs((train_acc[-1] - train_acc[-2]) / (train_acc[-2] - train_acc[-3])))
#     #     except:
#     #         pass
#
#     loss_avg = sum(loss_locals_train) / len(loss_locals_train)
#     train_loss.append(loss_avg)
#     try:
#         temp_zero_rates = sum(local_zero_rates) / len(local_zero_rates)
#     except:
#         temp_zero_rates = sum(local_zero_rates)
#     update_zero_rate.append(temp_zero_rates)
#
#     #     writer.add_scalar("Loss/train", loss, epoch)
#     #     writer.flush()
#     print('Round {} costs time: {:.2f}s| Train Acc.: {:.2%}| '
#           'Test Acc.{:.2%}| Train loss: {:.4f}| Test loss: {:.4f}| '
#           'Down Rate is {:.3%}| Up Rate{:.3%}'
#           ' Floating agg {}'.format(
#         epoch,
#         elapsed,
#         avg_acc_train,
#         test_acc[-1],
#         loss_avg,
#         test_loss[-1],
#         cr,
#         temp_zero_rates,
#         glob_agg_num
#     ))
#
#     current_lr = current_learning_rate(epoch, current_lr, args)
#
# his_dict = {
#     'train_loss': train_loss,
#     'train_accuracy': train_acc,
#     'test_loss': test_loss,
#     'test_correct': test_acc,
#     'train_time': train_time,
#     'glob_zero_rates': comp_rate,
#     'local_zero_rates': update_zero_rate,
#
# }
#
# os.makedirs('./his/', exist_ok=True)
# with open('./his/{}.json'.format(args.his), 'w+') as f:
#     json.dump(his_dict, f, indent=2)
