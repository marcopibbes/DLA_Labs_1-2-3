import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.nn import Dropout
from torch.nn import Flatten
from torch.nn import Identity
from torch.nn import ModuleList
from torch.nn import AdaptiveAvgPool2d
from torch.nn import Conv2d
from torch.nn import ConvTranspose2d
from torch.nn import MaxPool2d
from torch.nn import AvgPool2d
from torchvision.models.resnet import BasicBlock, Bottleneck


'''Why this implementation? 

Modularity and flexibility:
Each layer is defined by a dictionary of parameters, allowing you to build different architectures without rewriting code.

Reuse and customization:
The code supports custom weight initialization, different activation types, batch norm, dropout, pooling, and various convolution configurations.

Residual network support:
ResCNN allows you to build ResNet-like architectures, which are essential for deep and performant CNNs.

Easy integration with MLPs:
CNNs often extract features, while MLP classifies or regresses; here you can easily combine both.

Use of native PyTorch functions:
The use of nn.Sequential, nn.ModuleList, and PyTorch layers allows for automatic optimizations and compatibility with other modules.
'''
class CNN(nn.Module):
    def __init__(
        self,
        input_shape,
        conv_layers,
        mlp = None
        ):
        super().__init__()
        self.input_shape = input_shape
        self.conv_layers = conv_layers

        self.blocks = self._build_blocks()
        
        if mlp:
            self.blocks.append(mlp)


    ''' A sequential module is built for each convolutional layer with:

Configurable 2D convolution (kernel, stride, padding, etc.).
Customizable weight initialization.
Batch normalization to stabilize and accelerate training.
Activation (default ReLU).
2D dropout for regularization (at spatial level).
Optional pooling to reduce spatial dimension (e.g. MaxPool2d or AvgPool2d).
The number of input channels for the next layer is updated.

This structure is flexible and allows to easily define CNNs of different shapes with variable parameters.'''
       
    def _build_blocks(self):
        blocks = nn.ModuleList()

        in_channels = self.input_shape[0]
        for layer_params in self.conv_layers:
            layers = []
            
            assert "out_channels" in layer_params
            out_channels = layer_params["out_channels"]
            
            kernel = layer_params.get("kernel", 3)
            stride = layer_params.get('stride', 1)
            padding = layer_params.get('padding', 0)
            dilation = layer_params.get('dilation', 1)
            groups = layer_params.get('groups', 1)
            bias = layer_params.get("bias", True)

            init_func = layer_params.get("init_func", None)
            init_func_params = layer_params.get("init_func_params", {})

            act_func = layer_params.get("act_func", nn.ReLU)
            act_func_params = layer_params.get("act_func_params", {})

            batch_norm = layer_params.get("use_batch_norm", False)
            batch_norm_params = layer_params.get("batch_norm_params", {})
            bias = bias and not batch_norm

            dropout = layer_params.get("dropout", 0.0)

            pool = layer_params.get("pool", None)
            pool_kernel = layer_params.get("pool_kernel", kernel)
            pool_stride = layer_params.get("pool_stride", stride)
            pool_padding = layer_params.get("pool_padding", padding)
            pool_ceil_mode = layer_params.get("pool_ceil_mode", False)
            
            pool_params = layer_params.get("pool_params", {})
            if "kernel_size" not in pool_params or pool_params["kernel_size"] is None:
                pool_params["kernel_size"] = pool_kernel
            if "stride" not in pool_params or pool_params["stride"] is None:
                pool_params["stride"] = pool_stride
            if "padding" not in pool_params or pool_params["padding"] is None:
                pool_params["padding"] = pool_padding
            if "ceil_mode" not in pool_params or pool_params["ceil_mode"] is None:
                pool_params["ceil_mode"] = pool_ceil_mode
            
            conv = nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel,
                stride = stride,
                padding = padding,
                dilation = dilation,
                groups = groups,
                bias = bias
            )

            layers.append(conv)

            if init_func:
                init_func(layers[-1].weight, **init_func_params)

            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels, **batch_norm_params))

            if act_func:
                layers.append(act_func(**act_func_params))

            if dropout > 0.0:
                layers.append(nn.Dropout2d(p = dropout))  

            if pool:
                layers.append(pool(**pool_params))            

            blocks.append(nn.Sequential(*layers)) 

            in_channels = out_channels

        return blocks
    

    '''Calculates the final size of the output tensor after passing a sample tensor of batch size 1 through all blocks.
    Useful to know how many output elements (neurons) to connect a MLP or fully connected layer.'''

    def get_output_size(self):
        x = torch.zeros(1, *self.input_shape)
        
        for block in self.blocks:
            x = block(x)
        
        return x.numel()


    '''Allows to attach (add) a fully connected (MLP) module to the end of the CNN.
    Useful for creating hybrid CNN + MLP architectures (e.g. CNN for feature extraction + MLP for classification).'''

    def attach_mlp(self, mlp):
        self.blocks.append(mlp)


    '''Passes the input x through all convolutional blocks (and optionally the MLP).
    Returns the final output.'''
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

##############################################################################


'''Implements a CNN with residual blocks (like ResNet).

res_layers defines the parameters of each residual block.
block_type can be either BasicBlock or Bottleneck (standard residual block types).
An AdaptiveAvgPool2d is inserted to spatially aggregate the output into a fixed size (1x1).

A final MLP can be optionally added.'''
class ResCNN(nn.Module):
    def __init__(
        self,
        input_shape,
        res_layers,
        mlp=None,
        block_type=BasicBlock,
        base_width=64,
        norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        self.input_shape = input_shape
        self.res_layers = res_layers
        self.block_type = block_type
        self.base_width = base_width
        self.norm_layer = norm_layer

        self.blocks = self._build_blocks()

        self.blocks.append(nn.AdaptiveAvgPool2d((1, 1)))

        if mlp:
            self.blocks.append(mlp)




    '''For each residual layer:
Check if a downsample is needed (e.g. when dimension changes or stride > 1), using a 1x1 conv + batch norm to align dimensions.
Build a residual block (e.g. BasicBlock or Bottleneck) with the passed parameters.
Optionally add 2D dropout and pooling.
Update input channels for the next block considering the expansion of the residual block.

This structure replicates the flexibility and power of ResNets, which use residual jumps to improve the training of very deep CNNs.'''
    def _build_blocks(self):
        blocks = nn.ModuleList()
        in_channels = self.input_shape[0]

        for layer_params in self.res_layers:
            out_channels = layer_params["out_channels"]
            stride = layer_params.get('stride', 1)

            downsample = None
            if stride != 1 or in_channels != out_channels * self.block_type.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * self.block_type.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    self.norm_layer(out_channels * self.block_type.expansion)
                )

            block = self.block_type(
                inplanes=in_channels,
                planes=out_channels,
                stride=stride,
                downsample=downsample,
                groups=layer_params.get('groups', 1),
                base_width=self.base_width,
                dilation=layer_params.get('dilation', 1),
                norm_layer=self.norm_layer
            )

            layers = [block]

            if layer_params.get("dropout", 0.0) > 0.0:
                layers.append(nn.Dropout2d(p=layer_params["dropout"]))

            if "pool" in layer_params and layer_params["pool"] is not None:
                pool_cls = layer_params["pool"]
                pool_params = layer_params.get("pool_params", {
                    "kernel_size": layer_params.get("pool_kernel", 2),
                    "stride": layer_params.get("pool_stride", 2),
                    "padding": layer_params.get("pool_padding", 0)
                })
                layers.append(pool_cls(**pool_params))

            blocks.append(nn.Sequential(*layers))
            in_channels = out_channels * self.block_type.expansion

        return blocks

    def get_output_size(self):
        x = torch.zeros(1, *self.input_shape)
        for block in self.blocks:
            x = block(x)
        return x.numel()

    def attach_mlp(self, mlp):
        self.blocks.append(mlp)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x