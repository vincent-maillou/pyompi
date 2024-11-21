# Copyright 2023-2024 ETH Zurich. All rights reserved.

from warnings import warn

from numpy.typing import ArrayLike

from pyompi.__about__ import __version__

# Check if CuPy is available
CUPY_AVAILABLE = False
NCCL_AVAILABLE = False
try:
    import cupy as xp

    # Check if cupy is actually working. This could still raise
    # a cudaErrorInsufficientDriver error or something.
    xp.abs(1)

    CUPY_AVAILABLE = True

    # Check if NCCL is available
    try:
        import cupy.cuda.nccl as nccl

        # Check if NCCL is actually working
        nccl.get_version()

        NCCL_AVAILABLE = True
    except ImportError as e:
        warn(f"No 'NCCL' backend detected. ({e})")
except ImportError as e:
    warn(f"No 'CuPy' backend detected. ({e})")
    import numpy as xp

# Check if MPI is available
MPI_AVAILABLE = False
MPI_GPU_AWARE = False
try:
    from mpi4py import MPI

    MPI_AVAILABLE = True

    # If MPI is available and CuPy working, check if MPI is GPU aware
    if CUPY_AVAILABLE:
        try:
            info = MPI.COMM_WORLD.Get_info()
            gpu_aware = (
                "gpu" in info.get("mpi_memory_alloc_kinds", "")
                or "CRAY MPICH" in MPI.Get_library_version()
            )
            MPI_GPU_AWARE = MPI.COMM_WORLD.allreduce(gpu_aware, op=MPI.LAND)
        except Exception as e:
            warn(f"Could not determine if MPI is GPU aware, considered as not. ({e})")

except ImportError as e:
    warn(f"No 'MPI' backend detected. ({e})")




__all__ = [
    "__version__",
    "ArrayLike",
    "xp",
    "CUPY_AVAILABLE",
    "MPI_AVAILABLE",
    "MPI_GPU_AWARE",
    "NCCL_AVAILABLE",
]