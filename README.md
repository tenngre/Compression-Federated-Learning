# Compression Federated Learning Training

This git is a compression Federated Learning (FL) training framework. 
The compression method is following the paper "Neural Networks Weights Quantization: Target None-retraining Ternary (TNT)".
*[Paper Link](https://www.semanticscholar.org/paper/Neural-Networks-Weights-Quantization%3A-Target-(TNT)-Zhang-Zhu/8c826bc32fbc1f90d9e29649b61cf949241b588a)*

# RUN

This framework provides two ways of FL training, which are normal training and TNT training. The Normal training is without model compression and the TNT training is inverse.
Here are two examples for the two training ways, respectively.

1. TNT training RUN: `python tnt_fl_train_noniid.py  --tnt_uoload True`
2. Normal training RUN: `python tnt_fl_train_noniid.py`

# Model

All models are in the model directory, and a new model should be registered before training it.
You have to add the "register function" in front of the model class, an example is provided in VGG.py and Alex.py
Further, if you want to try other models do not exist in the model directory, you can place the model file in the models' directory file and register it.
Moreover, you can follow the vgg.py file to make a ternary model using TNTConv2d, TNTBatchNorm2d, and TNTLinear for ternary converting convolution layer, BatchNormal layer,
and Linear layer.

# Dataset

Currently, the framework only provide MNIST and CIFAR10 dataset in iid.

# Training config

All training parameters can be found in "config" in the "tnt_fl_train_noniid.py" file. Please try different parameters for traing different models.TNTBatchNorm2d


# Prerequisites

* python 3.6+
* pytorch 1.10.+