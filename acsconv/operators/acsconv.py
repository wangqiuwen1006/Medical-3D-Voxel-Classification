import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from collections import OrderedDict

from ..utils import _to_triple, _triple_same, _pair_same

from .base_acsconv import _ACSConv

class ACSConv(_ACSConv):
    """
    Vallina ACS Convolution
    
    Args:
        acs_kernel_split: optional, equally spit if not specified.

        Other arguments are the same as torch.nn.Conv3d.
    Examples:
        >>> import ACSConv
        >>> x = torch.rand(batch_size, 3, D, H, W)
        >>> conv = ACSConv(3, 10, kernel_size=3, padding=1)
        >>> out = conv(x)

        >>> conv = ACSConv(3, 10, acs_kernel_split=(4, 3, 3))
        >>> out = conv(x)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, acs_kernel_split=None, 
                 bias=True, padding_mode='zeros'):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, 0, groups, bias, padding_mode)
        if acs_kernel_split is None:
            if self.out_channels%3==0:
                self.acs_kernel_split = (self.out_channels//3,self.out_channels//3,self.out_channels//3)
            if self.out_channels%3==1:
                self.acs_kernel_split = (self.out_channels//3+1,self.out_channels//3,self.out_channels//3)
            if self.out_channels%3==2:
                self.acs_kernel_split = (self.out_channels//3+1,self.out_channels//3+1,self.out_channels//3)
        else:
            self.acs_kernel_split = acs_kernel_split

    def conv3D_output_shape_f(self,i, input_shape):
        """
        Calculate the original output size assuming the convolution is nn.Conv3d based on 
        input size, kernel size, dilation, padding and stride.
        """
        return math.floor((input_shape[i]-self.kernel_size[i]-(self.dilation[i]-1)*
                                         (self.kernel_size[i]-1)+2*self.padding[i])
                                        /self.stride[i])+1
    
    def forward(self, x):
        """
        Convolution forward function
        Divide the kernel into three parts on output channels based on acs_kernel_split, 
        and conduct convolution on three directions seperately. Bias is added at last.
        """
        
        B, C_in, *input_shape = x.shape
        conv3D_output_shape = (self.conv3D_output_shape_f(0, input_shape), 
                               self.conv3D_output_shape_f(1, input_shape), 
                               self.conv3D_output_shape_f(2, input_shape))
                
        weight_a = self.weight[0:self.acs_kernel_split[0]].unsqueeze(2)
        weight_c = self.weight[self.acs_kernel_split[0]:(self.acs_kernel_split[0]+self.acs_kernel_split[1])].unsqueeze(3)
        weight_s = self.weight[(self.acs_kernel_split[0]+self.acs_kernel_split[1]):].unsqueeze(4)
        f_out = []
        if self.acs_kernel_split[0]>0:
            a = F.conv3d(x if conv3D_output_shape[0]==input_shape[0] or 2*conv3D_output_shape[0]==input_shape[0] else F.pad(x, (0,0,0,0,self.padding[0],self.padding[0]),'constant',0)[:,:,
                                                self.kernel_size[0]//2:self.kernel_size[0]//2+(conv3D_output_shape[0]-1)*self.stride[0]+1,
                                                :,:], 
                                                weight=weight_a, bias=None, 
                                                stride=self.stride,
                                                padding=(0,self.padding[1],self.padding[2]),
                                                dilation=self.dilation,
                                                groups=self.groups)                
            f_out.append(a)
        if self.acs_kernel_split[1]>0:
            c = F.conv3d(x if conv3D_output_shape[1]==input_shape[1] or 2*conv3D_output_shape[1]==input_shape[1] else F.pad(x, (0,0,self.padding[1],self.padding[1]),'constant',0)[:,:,:,
                                                self.kernel_size[1]//2:self.kernel_size[1]//2+self.stride[1]*(conv3D_output_shape[1]-1)+1,
                                                :], 
                                                weight=weight_c, bias=None,                                     
                                                stride=self.stride,
                                                padding=(self.padding[0],0,self.padding[2]),
                                                dilation=self.dilation,
                                                groups=self.groups)
            f_out.append(c)
        if self.acs_kernel_split[2]>0:
            s = F.conv3d(x if conv3D_output_shape[2]==input_shape[2] or 2*conv3D_output_shape[2]==input_shape[2] else F.pad(x, (self.padding[2],self.padding[2]),'constant',0)[:,:,:,:,
                                                self.kernel_size[2]//2:self.kernel_size[2]//2+self.stride[2]*(conv3D_output_shape[2]-1)+1
                                                ], 
                                                weight=weight_s, 
                                                bias=None, 
                                                stride=self.stride,
                                                padding=(self.padding[0],self.padding[1],0),
                                                dilation=self.dilation,
                                                groups=self.groups)
            f_out.append(s)
        f = torch.cat(f_out, dim=1)
        if self.bias is not None:
            f += self.bias.view(1,self.out_channels,1,1,1)
        return f

    def extra_repr(self):
        s = super().extra_repr() + ', acs_kernel_split={acs_kernel_split}'
        return s.format(**self.__dict__)