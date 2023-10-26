import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
from pdb import set_trace

from torch.nn.utils import weight_norm, spectral_norm
from torch.nn.init import kaiming_uniform_,uniform_,xavier_uniform_,normal_
from torch.nn.modules.utils import _pair
from collections import OrderedDict

from .feature_extractor import get_layer_shapes

# ===================================================================
#  Upsampling
# ===================================================================

def init_linear(m, act_func=None, init='auto', bias_std=0.01):
    if getattr(m,'bias',None) is not None and bias_std is not None: normal_(m.bias, 0, bias_std)
    if init=='auto':
        if act_func in (F.relu_,F.leaky_relu_): init = kaiming_uniform_
        else: init = getattr(act_func.__class__, '__default_init__', None)
        if init is None: init = getattr(act_func, '__default_init__', None)
    if init is not None: init(m.weight)

def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function"
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(x.new_zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    return k.contiguous().view([nf,ni,h,w]).transpose(0, 1)

class PixelShuffle_ICNR(nn.Sequential):
    "via fastai: Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`."
    def __init__(self, ni, nf=None, scale=2, blur=False, norm_type='Weight', act_cls=nn.ReLU):
        super().__init__()
        nf = ni if nf is None else nf
        conv = nn.Conv2d(ni, nf*(scale**2), 1)
        if norm_type == 'Weight': conv = weight_norm(conv)
        act = nn.ReLU(inplace=False)
        init_linear(conv, act, init='auto', bias_std=0)
        layers = [conv, act, nn.PixelShuffle(scale)]
        layers[0].weight.data.copy_(icnr_init(layers[0].weight.data))
        if blur: layers += [nn.ReplicationPad2d((1,0,1,0)), nn.AvgPool2d(2, stride=1)]
        super().__init__(*layers)  

class AdaptiveUpsample(nn.Module): 
    valid_upsample = ['UpsampleBilinear', 'UpsampleBicubic', 'AdaptiveAvgPool2d', 'ConvTranspose2d', 'PixelShuffle_ICNR']
    def __init__(self, upsample_mode='AdaptiveAvgPool2d'):
        assert upsample_mode in self.valid_upsample, f"`upsample_mode` should be in {self.valid_upsample}, received {upsample_mode}"
        super().__init__()
        self.upsample_mode = upsample_mode
    
    def forward(self, x, out_size):
        if self.upsample_mode == "UpsampleBilinear":
            x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=True)
        elif self.upsample_mode == "UpsampleBicubic":
            x = F.interpolate(x, size=out_size, mode='bicubic', align_corners=True)
        elif self.upsample_mode == "AdaptiveAvgPool2d":
            x = F.adaptive_avg_pool2d(x, out_size)

        return x
    
    def __repr__(self):
        return f"{self.__class__.__name__}(upsample_mode='{self.upsample_mode}')"
    
class UpsampleBlock(nn.Sequential):
    valid_upsample = ['UpsampleBilinear', 'UpsampleBicubic', 'AdaptiveAvgPool2d', 'ConvTranspose2d', 'PixelShuffle_ICNR']
    
    def __init__(self, in_channels, out_channels, in_shape, out_shape, upsample_mode='PixelShuffle_ICNR'):
        assert upsample_mode in self.valid_upsample, f"`upsample_mode` should be in {self.valid_upsample}, received {upsample_mode}"
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.upsample_mode = upsample_mode
        
        if self.upsample_mode in ['UpsampleBilinear', 'UpsampleBicubic', 'AdaptiveAvgPool2d']:
            layers = OrderedDict([
                ('upsample', AdaptiveUpsample(upsample_mode=upsample_mode)),
                ('conv1x1', nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1,1), bias=False))
            ])      
            
        elif self.upsample_mode == 'ConvTranspose2d':
            layers = OrderedDict([])
            num_steps = round(self.out_shape/self.in_shape/2) # number of steps needed to increase by 2s to reach desired out_shape
            ni_ = self.in_channels # in_channels starts as # channels in feedback layer
            for c in range(num_steps):
                nout_ = self.out_channels # could be fancier and scale this
                layers[f'upconv{c}'] =  nn.Sequential(OrderedDict([
                    ('upconv2x2', nn.ConvTranspose2d(ni_, nout_, kernel_size=(2,2), stride=(2,2), padding=0, output_padding=0)),
                    ('relu', nn.ReLU(inplace=True))
                ]))
                ni_ = out_channels # in_channels for the next upsampling step is 

            layers['avg_pool'] = AdaptiveUpsample(upsample_mode='UpsampleBilinear')
            
        elif self.upsample_mode == 'PixelShuffle_ICNR':
            layers = OrderedDict([])
            num_steps = round(self.out_shape/self.in_shape/2) # number of steps needed to increase by 2s to reach desired out_shape
            ni_ = self.in_channels # in_channels starts as # channels in feedback layer
            for c in range(num_steps):
                nout_ = self.out_channels # could be fancier and scale this
                layers[f'upconv{c}'] = PixelShuffle_ICNR(ni_, nf=nout_, scale=2, blur=False, norm_type='Weight', act_cls=nn.ReLU)
                ni_ = out_channels # in_channels for the next upsampling step is 

            layers['avg_pool'] = AdaptiveUpsample(upsample_mode='UpsampleBilinear')
        
        super().__init__(layers)
        
    def forward(self, input, out_size=None):
        for module in self:
            if len(inspect.getfullargspec(module.forward).args)==3:
                input = module(input, out_size)
            else:
                input = module(input)
        return input
        
# ===================================================================
#  Softmax
# ===================================================================

class Softmax(nn.Module):
    '''
        softmax over channels ('C'), space ('HxW'), or both ('CxHxW')
    '''
    valid_dim = ['C', 'HxW', 'CxHxW']
    def __init__(self, dim):
        assert dim in self.valid_dim, f"`dim` should be in {self.valid_dim}, received {dim}"
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        bs,nc = x.shape[0:2]
        if self.dim == 'C':
            x = F.softmax(x, dim=1)
        elif self.dim == 'HxW':
            x = F.softmax(x.reshape(bs, nc, -1), dim=-1).reshape_as(x)
        elif self.dim == 'CxHxW':
            x = F.softmax(x.reshape(bs, -1), dim=-1).reshape_as(x)
        return x
    
    def __repr__(self):
        return f"{self.__class__.__name__}(dim='{self.dim}')"      
    
# ===================================================================
#  LRM Modules
# ===================================================================    

class FeedbackScale(nn.Module):
    valid_modes = ['tanh']
    def __init__(self, mode='tanh'):
        assert mode in self.valid_modes, f"Oops, mode must be in {self.valid_modes}, got {mode}"
        super().__init__()
        self.mode = mode
        
    def forward(self, x: Tensor, *args) -> Tensor:
        if self.mode == 'tanh':
            x = torch.tanh(x)
        return x
    
    def __repr__(self):
        return f"{self.__class__.__name__}(mode='{self.mode}')"

class MemoryFormatModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory_format = torch.contiguous_format
        
    def set_memory_format(self, memory_format):
        self.memory_format = memory_format

    def _apply(self, fn):
        '''
            When we call model.to(memory_format=...) we want to capture the memory_format 
            and apply it to the tensor `x` in the forward pass after it's unsqueezed to add 
            the spatial dimensions.
            
            This solution uses introspection on Python functions (specifically, looking into 
            the closure of the convert function). While this works for the current version of 
            PyTorch I'm using (1.13.1.post200), it does depend on internal details that might 
            change in future versions.             
        '''
        # The usual behavior
        super(MemoryFormatModule, self)._apply(fn)
        
        # Access closure variables
        if fn.__name__ == "convert":
            closure_vars = {k: v for k, v in zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__))}
            memory_format = closure_vars.get("convert_to_format", None)
            if memory_format is not None:
                self.set_memory_format(memory_format)
                
        return self
    
class AddSpatialDimension(MemoryFormatModule):
    def forward(self, x: Tensor, *args) -> Tensor:
        if len(x.shape) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
            x = x.to(memory_format=self.memory_format)
        return x
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
class ChannelNorm(nn.LayerNorm):
    def forward(self, x: Tensor, *args) -> Tensor:        
        if len(x.shape) == 4: 
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
        elif len(x.shape) == 2:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            raise ValueError(f'expected x.shape to be 2 or 4 numbers, got {x.shape}')
        return x
    
class AdaptiveFullstackNorm(nn.LayerNorm):
    def forward(self, x: Tensor, *args) -> Tensor:
        memory_format = torch.channels_last if x.is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format
        weight = F.interpolate(self.weight.unsqueeze(0), size=x.shape[-2:], mode='bilinear', align_corners=True).squeeze()
        bias = F.interpolate(self.bias.unsqueeze(0), size=x.shape[-2:], mode='bilinear', align_corners=True).squeeze()
        x = F.layer_norm(x, x.shape[-3:], weight, bias, self.eps)
        x = x.to(memory_format = memory_format)
        return x 
    
class NormSquashResize(nn.Module):
    norm_types = ['ChannelNorm', 'AdaptiveFullstackNorm']
    def __init__(self, in_channels, in_shape, out_shape, norm_type='ChannelNorm', scale_type='tanh', resize_type='UpsampleBilinear'):
        assert norm_type in self.norm_types, f"`norm_type` must be in {self.norm_types}, got {norm_type}"
        out_shape = _pair(out_shape)
        
        super().__init__()
        
        if in_shape == 1 or in_shape == _pair(1):
            self.norm = ChannelNorm(in_channels, elementwise_affine=True)
        elif norm_type == 'ChannelNorm':
            self.norm = ChannelNorm(in_channels, elementwise_affine=True)
        elif norm_type == 'AdaptiveFullstackNorm':
            self.norm = AdaptiveFullstackNorm((in_channels, *out_shape), elementwise_affine=True)
            
        self.squash = FeedbackScale(scale_type)
        
        if in_shape == 1 or in_shape == _pair(1):
            self.interp = AddSpatialDimension()
        else:
            self.interp = AdaptiveUpsample(upsample_mode=resize_type)
    
    def forward(self, x, out_size):
        x = self.norm(x)
        x = self.squash(x)
        x = self.interp(x, out_size)

        return x
    
class ResizeNormSquash(nn.Module):
    norm_types = ['ChannelNorm', 'AdaptiveFullstackNorm']
    def __init__(self, in_channels, in_shape, out_shape, norm_type='ChannelNorm', scale_type='tanh', resize_type='UpsampleBilinear'):
        assert norm_type in self.norm_types, f"`norm_type` must be in {self.norm_types}, got {norm_type}"
        out_shape = _pair(out_shape)
        
        super().__init__()
        
        if in_shape == 1 or in_shape == _pair(1):
            self.interp = AddSpatialDimension()
        else:
            self.interp = AdaptiveUpsample(upsample_mode=resize_type)
        
        if in_shape == 1 or in_shape == _pair(1):
            self.norm = ChannelNorm(in_channels, elementwise_affine=True)
        elif norm_type == 'ChannelNorm':
            self.norm = ChannelNorm(in_channels, elementwise_affine=True)
        elif norm_type == 'AdaptiveFullstackNorm':
            self.norm = AdaptiveFullstackNorm((in_channels, *out_shape), elementwise_affine=True)
                
        self.squash = FeedbackScale(scale_type)
    
    def forward(self, x, out_size):
        x = self.interp(x, out_size)
        x = self.norm(x)
        x = self.squash(x)
        
        return x   
    
class ModBlock(MemoryFormatModule):
    
    block_orders = ['norm-squash-resize', 'resize-norm-squash']            
             
    def __init__(self, name, in_channels, out_channels, in_shape, out_shape, block_order='norm-squash-resize', 
                 norm_type='ChannelNorm', resize_type='UpsampleBilinear'):
        assert block_order in self.block_orders, f"`block_order` must be in {self.block_orders}, got {block_order}"
        super().__init__()
        
        self.name = name
        
        if block_order == 'norm-squash-resize':
            self.rescale = NormSquashResize(in_channels, in_shape, out_shape, scale_type='tanh', 
                                            norm_type=norm_type, resize_type=resize_type)
        elif block_order == 'resize-norm-squash':
            self.rescale = ResizeNormSquash(in_channels, in_shape, out_shape, scale_type='tanh', 
                                            norm_type=norm_type, resize_type=resize_type)
        
        self.neg_scale = torch.nn.Parameter(torch.FloatTensor([1.0]))
        self.pos_scale = torch.nn.Parameter(torch.FloatTensor([1.0]))
        
        self.modulation = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), bias=True)
    
    def forward(self, x, target_size):
        '''
            Say we have conv4 feeding back to conv1.
            
            This takes the conv4 activation map (x),
            
            - rescale: normalize, tanh, resize
            - gamma: apply seprate gamma to neg and pos values
                     so we can learn how much inhib vs. excit to add
            - modulation: then use a conv1x1 to map from conv4 channels to conv1 channels
                
        '''
        # normalize, squash, and spatially interpolate the feedback activations
        x = self.rescale(x, target_size)
        
        # scale neg and pos separately to allow assymetric inhibition/facilitation
        # torch.where doesn't play well with Linear layer unsqueeze
        # x = torch.where(x < 0, x * self.neg_scale, x * self.pos_scale)
        neg_mask, pos_mask = x < 0, x >= 0        
        x = x * (neg_mask.float() * self.neg_scale + pos_mask.float() * self.pos_scale)
        x = x.to(memory_format=self.memory_format)        
        
        # finally, weight source channels to target channels to get final modulation signal
        x = self.modulation(x)

        if target_size==1:
            # squeeze out the spatial dimensions
            x = x.squeeze(-1).squeeze(-1)
        
        return x

class LongRangeModulation(nn.Sequential):
    def __init__(self, model, mod_target, mod_sources, img_size=224, 
                 mod_block_order='norm-squash-resize',
                 mod_norm_type='AdaptiveFullstackNorm', 
                 mod_resize_type='UpsampleBilinear'):
        
        self.targ_hooks = []
        self.mod_inputs = {}
        self.mod_hooks = []
        self.remove_mod_inputs = False
        
        self.name = f"{mod_target.replace('.','_')}_modulation"
        
        # first get the output shapes for mod_target and mod_sources
        layer_names = [mod_target] + mod_sources
        x = torch.rand(1, 3, *_pair(img_size))
        shapes = get_layer_shapes(model, layer_names, x)
        
        # get list of modules from model
        model_layers = dict([*model.named_modules()])
        
        # register a forward hook for the target_module
        # this is where we'll modulate the target_module's output
        target_module = model_layers[mod_target]
        self.targ_hooks += [target_module.register_forward_hook(self.forward_hook_target)]
        
        # register a backward hook to clear mod_inputs after gradients computed
        # self.targ_hooks += [target_module.register_full_backward_hook(self.remove_state)]
        
        # iterate over modulation sources, adding a ModBlock for each
        layers = OrderedDict([])
        for source_layer_name in mod_sources:
            source_module = model_layers[source_layer_name]
            
            name=f'from_{source_layer_name.replace(".","_")}_to_{mod_target.replace(".","_")}'
            
            self.mod_hooks += [source_module.register_forward_hook(partial(self.hook_fn, name=name))]
            
            source_size = _pair(shapes[source_layer_name][2:]) or (1,1)
            
            target_size = _pair(shapes[mod_target][2:]) or (1,1)

            mod_block_params = dict(
                name=name,
                source_channels=shapes[source_layer_name][1],
                target_channels=shapes[mod_target][1],
                source_size=source_size,
                target_size=target_size,
                mod_block_order=mod_block_order,
                mod_norm_type=mod_norm_type,
                mod_resize_type=mod_resize_type
            )
            # print(mod_block_params)
            
            # ModBlock handles aligning the source activation to the target activation
            # Normalizing, Squashing, and Resizing, then conv1x1 source => target
            modblock = ModBlock(name=mod_block_params['name'],
                                in_channels = mod_block_params['source_channels'],
                                out_channels = mod_block_params['target_channels'],
                                in_shape = mod_block_params['source_size'],
                                out_shape = mod_block_params['target_size'],
                                block_order = mod_block_params['mod_block_order'],
                                norm_type = mod_block_params['mod_norm_type'],
                                resize_type = mod_block_params['mod_resize_type'])
            layers[name] = modblock
        
        # a few identity blocks to make hooking different states easier (or torchlens?)
        layers['pre_mod_output'] = nn.Identity()
        layers['total_mod'] = nn.Identity()
        layers['post_mod_output'] = nn.Identity()

        super().__init__(layers)
    
    def forward_hook_target(self, module, input, output):
        '''modulate target output'''        
        
        if len(self.mod_inputs)==0:
            return output
        
        # pass through identity module to retain pre-modulation output
        self.pre_mod_output(output)
        
        # we need to know current output size to adaptively resize in ModBlock
        target_size = output.shape[-2:] if len(output.shape)==4 else 1
        
        # calculate long-range modulation to apply to output (sum across sources)
        total_mod = torch.zeros_like(output)
        for module in self:
            if hasattr(module, 'name',) and module.name in self.mod_inputs:
                source_activation = self.mod_inputs[module.name]

                mod = module(source_activation, target_size=target_size)

                total_mod = total_mod + mod
            
            # pass through identity module so total_mod can be read out
            self.total_mod(total_mod)

        # apply the modulation (x = x + x * f)
        output = output + output * total_mod

        # pass through identity module modulated output can be readout
        self.post_mod_output(output)
        
        # activation
        output = F.relu(output, inplace=False)
        
        return output
    
    def hook_fn(self, module, input, output, name):
        self.mod_inputs[name] = output
       
    def forward(self, x):
        '''forward pass of lrm modules doesn't get called'''
        pass