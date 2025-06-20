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

'''Why this MLP & ResMLP implementation?

The code creates a highly configurable MLP with customizable bias, initializations, batch norms, activations, and dropouts.
ResMLP extends MLP to add residual skip connections, which are essential for training deeper and more stable models.
Modular design and use of normalized lists allows for flexibility and reusability.
Projection functions for skip connections solve problems of different sizes in residual inputs.
Using PyTorch with Sequential and ModuleList blocks allows for clean and easily expandable models.
Intelligently handles the use of bias when there is batch norm (avoiding redundancies).'''

class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes,
        #all the following params can be:
        #- None (taking the default value)
        #- a single value (all layers will use this value)
        #- a list of values (one for each layer, if smaller than layer sizes its filled with default values)
        use_bias = True, 
        init_funcs = None,
        init_funcs_params = None,
        use_batch_norm = False,
        batch_norm_params = None,
        act_funcs = None,
        act_funcs_params = None,
        dropouts = None
    ):   
        super().__init__()

        self.n_layers = len(layer_sizes)
        self.n_linear = self.n_layers - 1

        assert self.n_layers > 1, "at least two layer for input and output are required"

        self.layer_sizes = layer_sizes

        #prepare bias
        self.bias = self._normalize_arguments(use_bias, self.n_linear, True)

        #prepare inititialization functions
        self.initf = self._normalize_arguments(init_funcs, self.n_linear) 
        self.initfp = self._normalize_arguments(init_funcs_params, self.n_linear, {}) 

        #prepare batch normalization
        self.batch = self._normalize_arguments(use_batch_norm, self.n_linear, False)
        self.batchp = self._normalize_arguments(batch_norm_params, self.n_linear, {})

        #prepare activation functions params
        self.actf = self._normalize_arguments(act_funcs, self.n_linear)
        self.actfp = self._normalize_arguments(act_funcs_params, self.n_linear, {})

        #prepare dropouts
        self.dropout = self._normalize_arguments(dropouts, self.n_linear, 0.0)
        assert all(0.0 <= d <= 1.0 for d in self.dropout), "dropout must be between 0. and 1."

        self.flatten = nn.Flatten()

        self.blocks = self._build_blocks()



    '''
   Helper function to standardize parameters.

If x is None, creates a long list size with default values.
If x is a single value, creates a list with that value repeated size times.
If x is a list, extends it with default values ​​up to length size.
It is used to elegantly manage parameter flexibility.
    '''
    def _normalize_arguments(self, x, size, default = None):
        l = []
        if x is None: #we are not using "not x" couse if x is False and default value is True it would create [True] * size instead of [False] * size
            l = [default] * size
        elif not isinstance(x, list):
            l = [x] * size
        else:
            to_fill = size - len(x) 
            assert to_fill >= 0, "x cannot have length greater than size"
            l = x + [default] * to_fill
        return l


    
    '''
   Builds a block list (nn.ModuleList) for each linear layer.

For each layer:
Creates nn.Linear with its inputs/outputs.
Applies the initialization function (if defined).
Adds BatchNorm1d if required.
Adds the activation function (e.g. ReLU).
Adds the dropout.
All these steps are concatenated in a nn.Sequential.
Linear layers do not use bias if BatchNorm is active, to avoid redundancies.

Reason: create a modular structure, easily extendable and customizable.
    '''
    def _build_blocks(self):
        blocks = nn.ModuleList()

        for i in range(self.n_linear):
            layers = []
            
            #use bias only if there is not batch normalization, otherwise would be redoundant
            use_bias = self.bias[i] and not self.batch[i]

            layers.append(nn.Linear(in_features = self.layer_sizes[i], out_features = self.layer_sizes[i+1], bias = use_bias))
            if self.initf[i]: #if we have defined a function for weight initialization, apply it, otherwise use the default one from pytorch
                self.initf[i](layers[-1].weight, **self.initfp[i])

            #adding batch normalization
            if self.batch[i]:
                layers.append(nn.BatchNorm1d(self.layer_sizes[i+1], **self.batchp[i]))

            #adding activation function
            if self.actf[i]:
                layers.append(self.actf[i](**self.actfp[i]))

            #adding dropout
            if self.dropout[i] > 0.0:
                layers.append(nn.Dropout(p = self.dropout[i]))  

            blocks.append(nn.Sequential(*layers))

        return blocks
    

'''
First, if x has more than 2 dimensions (e.g. 4D images), flatten it with nn.Flatten.
Pass the input sequentially through all the constructed blocks.
Return the final output.
'''
def forward(self, x):
        if x.ndim > 2:
            x = self.flatten(x)

        for i in range(self.n_linear):
            x = self.blocks[i](x)

        return x

##############################################################################


'''ResMLP can receive an already instantiated mlp object or a layer_sizes list.
If it receives an existing MLP, it copies its parameters as default.
skip_connections: list of tuples (from_layer_idx, to_layer_idx) that define the residual connections.
If not specified, the default skip connection is from input (0) to output (n_linear - 1).
Each skip connection can have an associated projection function (projection_funcs), useful if the input and output sizes of the skip do not match.'''
class ResMLP(MLP):
    def __init__(
        self,
        mlp = None,
        layer_sizes = None,
        use_bias = True, 
        init_funcs = None,
        init_funcs_params = None,
        use_batch_norm = False,
        batch_norm_params = None,
        act_funcs = None,
        act_funcs_params = None,
        dropouts = None,
        skip_connections = None,
        projection_funcs = None,
        projection_funcs_params = None,    
    ):
        assert mlp or layer_sizes, "either mlp or layer_sizes must be provided"

        if mlp and isinstance(mlp, MLP):
            layer_sizes = mlp.layer_sizes or layer_sizes
            use_bias = mlp.bias or use_bias
            init_funcs = mlp.initf or init_funcs
            init_funcs_params = mlp.initfp or init_funcs_params
            use_batch_norm = mlp.batch or use_batch_norm
            batch_norm_params = mlp.batchp or batch_norm_params
            act_funcs = mlp.actf or act_funcs
            act_funcs_params = mlp.actfp or act_funcs_params
            dropouts = mlp.dropout or dropouts

        super().__init__(
            layer_sizes = layer_sizes,
            use_bias = use_bias,
            init_funcs = init_funcs,
            init_funcs_params = init_funcs_params,
            use_batch_norm = use_batch_norm,
            batch_norm_params = batch_norm_params,
            act_funcs = act_funcs,
            act_funcs_params = act_funcs_params,
            dropouts = dropouts
        )

        if not skip_connections: 
            skip_connections = [(0, self.n_linear - 1)] #skip from input to output by default

        '''Ensure skip connection indices are valid and consistent.

        from_idx must be less than or equal to to_idx to avoid loops or errors.'''

        assert isinstance(skip_connections, (list, tuple)), "skip_connections must be a list or tuple of (from, to) tuples"
        for from_idx, to_idx in skip_connections:
            assert 0 <= from_idx <= to_idx < self.n_linear, f"Invalid skip connection indices ({from_idx}, {to_idx}). Must satisfy 0 <= from <= to < {n_linear}"

        self.skip_connections = skip_connections
        n_skips = len(self.skip_connections)

        proj_funcs = self._normalize_arguments(projection_funcs, n_skips, None)
        proj_funcs_params = self._normalize_arguments(projection_funcs_params, n_skips, {})

        self.projections = nn.ModuleList()
        self.skip_indices = {}
        
        for idx, (from_idx, to_idx) in enumerate(self.skip_connections):
            self.skip_indices[(from_idx, to_idx)] = idx
            
            proj_func = proj_funcs[idx]
            proj_func_params = proj_funcs_params[idx]
            
            if proj_func:
                self.projections.append(proj_func(layer_sizes[from_idx], layer_sizes[to_idx], **proj_func_params))
            elif layer_sizes[from_idx] != layer_sizes[to_idx]:
                self.projections.append(nn.Linear(layer_sizes[from_idx], layer_sizes[to_idx], bias=False))
            else:
                self.projections.append(nn.Identity())


    '''Flattens input if multidimensional.

Stores the intermediate output of each layer for skip connections.
When it arrives at the target index to_idx of a skip connection, it sums the current output with the projected residual.
Returns the final output with the residual sums performed.
Reason: Residual connections help train deep networks by improving gradient flow and allowing to "skip" layers, a concept inspired by ResNets.'''    
    def forward(self, x):
        if x.ndim > 2:
            x = self.flatten(x)
        
        intermediate_outputs = {0: x}
        residuals_to_project = {}
        
        current_x = x
        for i in range(self.n_linear):
            #store residuals for later use in skip connections
            for from_idx, _ in self.skip_connections:
                if i == from_idx:
                    residuals_to_project[(from_idx, _)] = current_x
            
            #apply regular block
            current_x = self.blocks[i](current_x)
            
            #apply residual connections
            for from_idx, to_idx in self.skip_connections:
                if i == to_idx:
                    residual_input = residuals_to_project.get((from_idx, to_idx))
                    proj_idx = self.skip_indices.get((from_idx, to_idx))
                    projected_residual = self.projections[proj_idx](residual_input)
                    current_x = current_x + projected_residual
            
            if i + 1 < self.n_linear:
                intermediate_outputs[i + 1] = current_x
        
        return current_x