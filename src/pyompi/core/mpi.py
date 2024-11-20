# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

from pyompi.core import OMPI
from pyinla import MPI_AVAILABLE, ArrayLike, comm_rank

class MPI(OMPI):
    def print_msg(*args, **kwargs):
        """
        Print a message from a single process.

        Parameters:
        -----------
        *args:
            Variable length argument list.
        **kwargs:
            Arbitrary keyword arguments.
        """
        if comm_rank == 0:
            print(*args, **kwargs)

    def synchronize(comm=None):
        """
        Synchronize all processes within the given communication group.

        Parameters:
        -----------
        comm, optional:
            The communication group to synchronize. Default is MPI.COMM_WORLD.
        """
        if MPI_AVAILABLE:
            if comm is None:
                comm = MPI.COMM_WORLD
            comm.Barrier()

    def allreduce(
        sendbuf: ArrayLike,
        recvbuf: ArrayLike,
        op: str = "sum",
        comm=None,
    ):
        """
        Perform a reduction operation across all processes within the given communication group.

        Parameters:
        -----------
        sendbuf (ArrayLike):
            The buffer to send.
        recvbuf (ArrayLike):
            The buffer to receive.
        op ():
            The reduction operation.
        comm (MPI.Comm), optional:
            The communication group. Default is MPI.COMM_WORLD.
        """
        if MPI_AVAILABLE:
            if comm is None:
                comm = MPI.COMM_WORLD
            if op == "sum":
                comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)

    def bcast(
        data: ArrayLike,
        root: int = 0,
        comm=None,
    ):
        """
        Broadcast data from the root process to all other processes within the given communication group.

        Parameters:
        -----------
        data (ArrayLike):
            The data to broadcast.
        root (int), optional:
            The root process. Default is 0.
        comm (MPI.Comm), optional:
            The communication group. Default is MPI.COMM_WORLD.
        """
        if MPI_AVAILABLE:
            if comm is None:
                comm = MPI.COMM_WORLD
            comm.Bcast(data, root=root)