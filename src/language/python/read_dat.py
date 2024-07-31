import numpy as np
import scipy.sparse
path = "/home/zhuyangyang/Course/parallel_process/hpcg/build/bin/"

# Load A.dat and convert it to a sparse matrix
A_data = np.loadtxt(path + 'A-4096.dat')
# A_data = np.loadtxt(path + 'A-8.dat')
# rows = A_data[:, 0].astype(int) - 1  # Convert to 0-based indexing
# cols = A_data[:, 1].astype(int) - 1  # Convert to 0-based indexing
rows = A_data[:, 0].astype(int)    # Convert to 0-based indexing
cols = A_data[:, 1].astype(int)   # Convert to 0-based indexing
values = A_data[:, 2]
A = scipy.sparse.coo_matrix((values, (rows, cols)))

# Load x.dat
x = np.loadtxt(path +'x.dat')

# Load xexact.dat
xexact = np.loadtxt(path +'xexact.dat')

# Load b.dat
b = np.loadtxt(path +'b.dat')
