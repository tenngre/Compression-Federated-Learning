import torch
import torch.nn as nn
from ternay.convert_tnt import *
import copy
from models import register_network


@register_network('vgg_tnt')
class VGG_tnt(nn.Module):

    def __init__(self, nclass=100):
        super(VGG_tnt, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            TNTConv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1),
            TNTBatchNorm2d(128),
            nn.ReLU(),
            
            TNTConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            TNTBatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            TNTConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            TNTBatchNorm2d(256),
            nn.ReLU(),
            
            TNTConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            TNTBatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            TNTLinear(in_features=6400, out_features=1024),
            nn.ReLU(),
            TNTLinear(in_features=1024, out_features=nclass),
        )
        
    def get_tnt(self):
        w = copy.deepcopy(self.state_dict())
        for name, module in self.named_modules():
            if isinstance(module, TNTConv2d):
                w[name + str('.weight')] = KernelsCluster.apply(module.weight)
            if isinstance(module, TNTLinear):
                w[name + str('.weight')] = KernelsCluster.apply(module.weight)
        return w

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

class VGG_norm(nn.Module):

    def __init__(self, n_classes=10):
        super(VGG_norm, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=6400, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
#         print(x.shape)
        x = self.classifier(x)
        return x