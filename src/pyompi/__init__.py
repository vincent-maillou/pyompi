# Copyright 2023-2024 ETH Zurich. All rights reserved.

from warnings import warn

from numpy.typing import ArrayLike

from serinv.__about__ import __version__

# In any case, check if CuPy is available.
CUPY_AVAILABLE = False
try:
    import cupy as xp

    # Check if cupy is actually working. This could still raise
    # a cudaErrorInsufficientDriver error or something.
    xp.abs(1)

    CUPY_AVAILABLE = True
except ImportError as e:
    warn(f"No 'CuPy' backend detected. ({e})")


MPI_AVAILABLE = False
try:
    from mpi4py import MPI

    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()

    MPI_AVAILABLE = True
except ImportError as e:
    warn(f"No 'MPI' backend detected. ({e})")

    mpi_rank = 0
    mpi_size = 1

__all__ = [
    "__version__",
    "ArrayLike",
    "comm_rank",
    "comm_size",
    "CUPY_AVAILABLE",
    "MPI_AVAILABLE",
]