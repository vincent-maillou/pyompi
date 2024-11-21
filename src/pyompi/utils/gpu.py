# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import inspect
import warnings

import numpy as np
from numpy.typing import ArrayLike

from pyompi import xp

def get_array_module_name(arr: ArrayLike) -> str:
    """Given an array, returns the array's module name.

    This works for numpy even when cupy is not available.

    Parameters
    ----------
    arr : ArrayLike
        The array to check.

    Returns
    -------
    str
        The array module name used by the array.

    """
    submodule = inspect.getmodule(type(arr))
    return submodule.__name__.split(".")[0]

def get_host(in_arr: ArrayLike, out_arr: ArrayLike = None) -> ArrayLike:
    """Copy the given device array to the host.
    If no host array is given, a new one is created. Raise a warning 
    as it perform a D2H copy to a non pinned memory.

    Parameters
    ----------
    in_arr : ArrayLike
        The array to convert.

    Returns
    -------
    ArrayLike: optional
        The equivalent numpy array.

    """
    if get_array_module_name(in_arr) == "numpy":
            return in_arr

    if out_arr is None:
        warnings.warn("No host array given. A new one will be created and the copy will be performed to a non-pinned memory array.")
        return in_arr.get()
    else:
        in_arr.get(out=out_arr)

def get_device(in_arr: ArrayLike, out_arr: ArrayLike = None) -> ArrayLike:
    """Returns the device array of the given array.

    Parameters
    ----------
    arr : ArrayLike
        The array to convert.

    Returns
    -------
    ArrayLike
        The equivalent cupy array.

    """
    if get_array_module_name(in_arr) == "cupy":
        return in_arr
    
    if out_arr is None:
        return xp.asarray(in_arr)
    else:
        out_arr.set(in_arr)
    
def synchronize_current_stream() -> None:
    """Synchronizes the current stream if using cupy.

    Does nothing if using numpy.

    """
    if xp.__name__ == "cupy":
        xp.cuda.get_current_stream().synchronize()