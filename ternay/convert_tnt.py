from ternay.TNT import kernels_cluster
import torch.nn as nn
import torch.nn.functional as F
import torch


class KernelsCluster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return kernels_cluster(x)

    @staticmethod
    def backward(ctx, grad_output):
        return kernels_cluster(grad_output)

    
# class KernelsCluster2(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         return kernels_cluster(weights_f=x, channel=False)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output

    
class TNTConv2d(nn.Conv2d):
    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def forward(self, x):
        w = KernelsCluster.apply(self.weight) # .to(self.weight.device)
        if self.bias is not None:
            b = KernelsCluster.apply(self.bias)
        else:
            b = self.bias
        # print(w)
        y = self._conv_forward(x, w, b)
        
        return y
    
# class TNTFConv2d(nn.Conv2d):
#     def forward(self, x):
#         w = KernelsCluster2.apply(self.weight) # .to(self.weight.device)
#         # print(w)
#         y = self._conv_forward(x, w)
#         return y


class TNTLinear(nn.Linear):
    def forward(self, x):
        w = KernelsCluster.apply(self.weight)
        b = KernelsCluster.apply(self.bias)
        return F.linear(x, w, b)
    

class TNTBatchNorm2d(nn.BatchNorm2d):

    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            device=None,
            dtype=None
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        self.num_batches_tracked = self.num_batches_tracked + 1

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        w = KernelsCluster.apply(self.weight)
        # print(w)
        b = KernelsCluster.apply(self.bias)

        return F.batch_norm(
            input, self.running_mean, self.running_var, w, b,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)