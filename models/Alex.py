import torch
import torch.nn as nn
from ternay.convert_tnt import *
import copy


class AlexNet_tnt(nn.Module):
    def __init__(self, num_classes=10, tnt_state=False):
        super(AlexNet_tnt, self).__init__()
        self.features = nn.Sequential(
            TNTConv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            TNTConv2d(64, 192, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            TNTConv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            TNTConv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            TNTConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            TNTLinear(in_features=256 * 4 * 4, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            #             TNTLinear(4096, 4096),
            #             nn.ReLU(inplace=True),
            TNTLinear(256, num_classes),
        )

    def get_tnt(self):
        w = copy.deepcopy(self.state_dict())
        for name, module in self.named_modules():
            if isinstance(module, TNTConv2d):
                w[name + str('.weight')] = KernelsCluster.apply(module.weight)
            if isinstance(module, TNTLinear):
                w[name + str('.weight')] = KernelsCluster.apply(module.weight)
        return w

    def forward(self, x, tnt_state=False):
        for module in self.features:
            if isinstance(module, TNTConv2d):
                module.tnt = tnt_state
        for module in self.classifier:
            if isinstance(module, TNTLinear):
                module.tnt = tnt_state

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
