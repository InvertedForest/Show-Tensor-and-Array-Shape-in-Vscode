import numpy as np
import torch

def custom_repr_numpy(self):
    def get_first(j):
        _value = self
        for i in range(len(self.shape)-1):
            _value = _value[0]
        return _value[j]
    if self.shape[-1] == 0:
        stri = ''
    elif self.shape[-1] == 1:
        stri = '[' + str(get_first(0)) + ']'
    else:
        stri = '['*len(self.shape) + f'{get_first(0)}, {get_first(1)}' + '...'
    return f"{self.shape} array: {stri}"  
np.set_string_function(custom_repr_numpy, repr=True)

original_repr = torch.Tensor.__repr__
def custom_repr_torch(self):
    shape_str = str(tuple(self.shape))
    if shape_str[-2] == ',':
        shape_str = shape_str[:-2] + ')'
    dtype = str(self.dtype).split('torch.')[1]
    return f"{tuple(self.shape)} '{dtype}' '{self.device}' {original_repr(self)}"
torch.Tensor.__repr__ = custom_repr_torch