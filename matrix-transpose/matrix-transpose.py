import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    r, c = len(A), len(A[0])
    t = [[0] * r for _ in range(c)]
    for i in range(r):
        for j in range(c):
            t[j][i] = A[i][j]
    t = np.array(t)
    return t
    pass
