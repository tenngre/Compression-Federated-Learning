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
            TNTConv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1),
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
            TNTLinear(in_features=4096, out_features=1024),
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

    
@register_network('vgg_norm')
class VGG_norm(nn.Module):

    def __init__(self, nclass=10):
        super(VGG_norm, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1),
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
            nn.Linear(in_features=4096, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=nclass),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@register_network('net')
class Net(nn.Module):
    def __init__(self, nclass=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, nclass)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
