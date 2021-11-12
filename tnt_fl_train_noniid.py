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
parser.add_argument('--tnt_upload', action='store_true', help='uploading tnt weights')
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
dict_users_train, dict_users_test = cifar_extr_noniid(dataset_train,
                                                      dataset_test,
                                                      args.num_users,
                                                      args.n_class)

config = {
    # 'current_lr': args.lr,
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
        'alex_norm': AlexNet},

    'log_dir': 'results'
}

if __name__ == '__main__':

    if args.tnt_upload:
        print("YES")
        training.main_tnt_upload(config, args)
    else:
        print("NOOOOOOOOO")
        training.main_norm_upload(config, args)
