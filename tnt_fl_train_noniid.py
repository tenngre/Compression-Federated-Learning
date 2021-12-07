import configs
import os
import argparse
import random
import numpy as np
from scripts import training
import torch

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--num_users', default=1, type=int, )
parser.add_argument('--epochs', default=100, help='epoch', type=int)
parser.add_argument('--frac', default=1, type=int)
parser.add_argument('--local_bs', default=256, type=int)
parser.add_argument('--save', action='store_true', help='save model every 10 epoch')
parser.add_argument('--GPU', default=0, type=int)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--split', default='user')
parser.add_argument('--local_ep', default=1, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--bs', default=256, type=int)
parser.add_argument('--d_epoch', default=50, type=int)
parser.add_argument('--decay_r', default=0.1, type=float)
parser.add_argument('--tnt_upload', action='store_true', help='uploading tnt weights')
parser.add_argument('--weight_decay', default=0.0001, type=float)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--seed', default=80, type=int)
parser.add_argument('--model', default='alex_tnt', type=str)
parser.add_argument('--n_class', default=2, type=int, help='class number in each client')
parser.add_argument('--g_c', default=200, type=int, help='floating model communication epoch')
args = parser.parse_args()

args_dict = vars(args)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


config = {
    # 'current_lr': args.lr,
    'acc_rate': 0.5,
    'best_acc': 0,
    'glob_agg_num': 0,
    'device': device,
    # dataset
    'dataset': args.dataset,
    'dataset_kwargs': {
        'resize': 256 if args.dataset in ['nuswide'] else 0,
        'crop': 0,
        'norm': 2,  # 3 for mnist only; 2 for cifar10
        'evaluation_protocol': 2,  # only affect cifar10
        'reset': True,
        'separate_multiclass': False,
    },

    # FL global items
    'epochs': args.epochs,
    'save_interval': 20,
    'eval_interval': 1,
    'bs': args.bs,  # for testing
    'train_set': 0,
    'test_set': 0,
    'current_lr': args.lr,
    'seed': args.seed,

    # FL client
    'client_num': args.num_users,
    'reset_index': True,
    'n_class': args.n_class,
    'client_frac': args.frac,
    'tnt_upload': args.tnt_upload,
    'local_bs': args.local_bs,
    'local_ep': args.local_ep,
    'd_epoch': args.d_epoch,

    'weights_decay_inter': 30,
    'scheduler': 'cos',  # step, mstep, cos
    'scheduler_kwargs': {
        'step_size': int(args.epochs * 0.8),
        'gamma': 0.1,
        'milestones': '0.5,0.75'
    },

    'optima': 'sgd',  # sgd, adam
    'optima_kwargs': {
        'lr': args.lr,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'nesterov': False,
        'betas': (0.9, 0.999)
    },

    'model_name': args.model,
    'arch_kwargs': {
        'nclass': 0,  # will be updated below
        # 'pretrained': True,
        # 'freeze_weight': False,
    },
    'log_dir': 'results',
    'save': args.save,
    'tag': 'TNT' if args.tnt_upload else 'Norm'
}

config['arch_kwargs']['nclass'] = configs.nclass(config)
# config['R'] = configs.R(config)

logdir = (f'./{config["model_name"]}{config["arch_kwargs"]["nclass"]}_'
          f'{config["dataset"]}_{config["dataset_kwargs"]["evaluation_protocol"]}_'
          f'{config["epochs"]}')

if config['tag'] != '':
    logdir += f'/{config["tag"]}_{config["seed"]}_'
else:
    logdir += f'/{config["seed"]}_'

# make sure no overwrite problem
count = 0
orig_logdir = logdir
logdir = orig_logdir + f'{count:03d}'

while os.path.isdir(logdir):
    count += 1
    logdir = orig_logdir + f'{count:03d}'

config['logdir'] = logdir

count = 0
orig_logdir = logdir
logdir = orig_logdir + f'{count:03d}'

if __name__ == '__main__':
    random_seed(config['seed'])
    if config['tnt_upload']:
        print('YES')
        training.main_tnt_upload(config)
    else:
        print('NO')
        training.main_norm_upload(config)
