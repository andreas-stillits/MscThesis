from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    data = np.arange(size, dtype=np.int64)  # one item per rank
else:
    data = None

my_item = comm.scatter(data, root=0)
result  = my_item**2
all_sq  = comm.gather(result, root=0)

if rank == 0:
    print("input:", np.arange(size))
    print("squared:", np.array(all_sq))

