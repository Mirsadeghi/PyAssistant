import torch
import numpy as np
def get_mem():
    t = torch.cuda.get_device_properties(0).total_memory
    c = torch.cuda.memory_reserved(0)
    # c = torch.cuda.memory_cached(0)
    a = torch.cuda.memory_allocated(0)
    f = c-a  # free inside cache
    t /= 1024**3
    f /= 1024**3
    return t, f


def torch2numpy(x):
    return x.detach().cpu().numpy()

def isClass(data, dtype):
    name = str(type(data))
    if name.__contains__('str'):
        dtype_= str
    elif name.__contains__('list'):
        dtype_ = list
    elif name.__contains__('numpy.ndarray'):
        dtype_ = np.ndarray
    elif name.__contains__('torch.Tensor'):
        dtype_ = torch.Tensor
    else:
        dtype_ = None

    if dtype_ == dtype:
        return True
    else:
        return False