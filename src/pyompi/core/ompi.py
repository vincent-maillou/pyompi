# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

from abc import ABC, abstractmethod

from pyompi import ArrayLike

class OMPI(ABC):
    @abstractmethod
    def print_msg(*args, **kwargs):
        """Print a message from a single process.

        Parameters:
        -----------
        *args:
            Variable length argument list.
        **kwargs:
            Arbitrary keyword arguments.
        """
        ...

    @abstractmethod
    def synchronize():
        """Synchronize all processes within the given communication group.

        Parameters:
        -----------
        comm, optional:
            The communication group to synchronize. Default is MPI.COMM_WORLD.
        """
        ...
        
    @abstractmethod
    def allreduce(
        sendbuf: ArrayLike,
        recvbuf: ArrayLike,
        op: str = "sum",
        comm=None,
    ):
        """Perform a reduction operation across all processes within the given communication group.

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
        ...

    @abstractmethod
    def allgather(
        sendbuf: ArrayLike,
        recvbuf: ArrayLike,
        comm=None,
    ):
        """Gather data from all processes within the given communication group.

        Parameters:
        -----------
        sendbuf (ArrayLike):
            The buffer to send.
        recvbuf (ArrayLike):
            The buffer to receive.
        comm (MPI.Comm), optional:
            The communication group. Default is MPI.COMM_WORLD.
        """
        ...

    @abstractmethod
    def alltoall(
        sendbuf: ArrayLike,
        recvbuf: ArrayLike,
        comm=None,
    ):
        """Distribute data from all processes within the given communication group.

        Parameters:
        -----------
        sendbuf (ArrayLike):
            The buffer to send.
        recvbuf (ArrayLike):
            The buffer to receive.
        comm (MPI.Comm), optional:
            The communication group. Default is MPI.COMM_WORLD.
        """
        ...

    @abstractmethod
    def bcast(
        data: ArrayLike,
        root: int = 0,
        comm=None,
    ):
        """Broadcast data from the root process to all other processes within the given communication group.

        Parameters:
        -----------
        data (ArrayLike):
            The data to broadcast.
        root (int), optional:
            The root process. Default is 0.
        comm (MPI.Comm), optional:
            The communication group. Default is MPI.COMM_WORLD.
        """
        ...