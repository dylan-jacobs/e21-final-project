import numpy as np

def forward_elimination(aug):
    # A is a square matrix of size N x N,
    # e.g., [[1,2,3],[1,4,5],[0,1,2]]
    # b is a column vector of size N x 1
    # e.g., [[-1],[5],[7]]
    # Function returns augmented matrix [A | b]
    
    for i in range(0, len(aug)):
        for j in range(0, i):
            aug[i, :] = aug[i, :] - (aug[i, j]/aug[j, j])*aug[j, :]
    return aug

def backward_substitution(Ab):
    # Input is a N x (N+1) matrix
    # output is a column vector of size N x 1

    # Initialize your 'x' vector that will contain your solution
    N = np.shape(Ab)[0]
    xvals = np.zeros((N,1))
    for i in range(N-1, -1, -1):
        xvals[i] = (1/Ab[i, i])*(Ab[i, -1] - sum([Ab[i, j]*xvals[j] for j in range(i+1, N)]))

    return xvals

def nullspace(A):
    aug = forward_elimination(A)
    nullspace = backward_substitution(aug)
    return nullspace