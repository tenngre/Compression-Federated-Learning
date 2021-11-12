'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ternay.convert_tnt import *
import copy


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = TNTConv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False, groups=1)
        self.bn1 = TNTBatchNorm2d(planes)
        self.conv2 = TNTConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False, groups=1)
        self.bn2 = TNTBatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                TNTConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, groups=1),
                TNTBatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = TNTConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False, groups=1)
        self.bn1 = TNTBatchNorm2d(planes)
        self.conv2 = TNTConv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False, groups=1)
        self.bn2 = TNTBatchNorm2d(planes)
        self.conv3 = TNTConv2d(planes, self.expansion *
                               planes, kernel_size=1, stride=1, padding=0, bias=False, groups=1)
        self.bn3 = TNTBatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                TNTConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, padding=0, bias=False, groups=1),
                TNTBatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, tnt_state=False):
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = TNTConv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = TNTBatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = TNTLinear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, is_tnt=False):
        for module in self.modules():
            if isinstance(module, TNTConv2d):
                module.tnt = is_tnt
            if isinstance(module, TNTLinear):
                module.tnt = is_tnt

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def get_tnt(self):
        # get ternary type weights
        w = copy.deepcopy(self.state_dict())
        for name, module in self.named_modules():
            if isinstance(module, TNTConv2d):
                w[name + str('.weight')] = KernelsCluster.apply(module.weight)
            if isinstance(module, TNTLinear):
                w[name + str('.weight')] = KernelsCluster.apply(module.weight)
        return w


#------------------------------


def ResNet_TNT18(num_class):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_class)


def ResNet_TNT34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet_TNT50(num_class):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_class)


def ResNet_TNT101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet_TNT152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
