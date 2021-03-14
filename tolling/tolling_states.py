# states of the tolling model

import numpy as np
import pycuda.gpuarray as gpa

from enum    import Enum
from typing import Union

from pycuda.gpuarray import GPUArray


class TollingState(Enum):
    """ State that the power plant can be in.
    """

    NOT_RUNNING  = 1
    MIN_DISPATCH = 2
    MAX_DISPATCH = 3


class BlockStates(Enum):
    PEAK    = 1
    OFFPEAK = 2


def const_array(size : int, value, dtype_ : type = bool, cuda_ind : bool = False) -> Union[np.ndarray, GPUArray]:
    """ Returns a bool array of size size, with all values set to value.

    :param size: size of the array
    :param value: value the array is set to, can be float, short, etc.
    :param dtype_: type of the array to be generated
    :param cuda_ind: indicator for cuda.
    :returns: array of size size and value set to value, either np.array or gpu array
    """

    res = (np.empty if not cuda_ind else gpa.empty)(size, dtype=dtype_)
    res.fill(value)

    return res
