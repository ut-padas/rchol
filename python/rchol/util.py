import numpy as np
from numpy.linalg import norm

import scipy.sparse as sp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve_triangular

def laplace_3d(n):
    d0 = 2*np.ones(n);
    d1 = -1*np.ones(n-1);
    D = diags([d1,d0,d1], [-1,0,1])
    I = sp.identity(n)

    A1 = sp.kron(D,I)
    A2 = sp.kron(I,D)
    A = sp.kron(A1,I)+sp.kron(I,A1)+sp.kron(I,A2)
    A = A.tocsr(copy=True)
    A.eliminate_zeros()
    return A


def sddm_to_laplacian(M):
    one_row = -np.squeeze(np.asarray(M.sum(axis=0)))
    np.where(np.abs(one_row) < 1e-9, 0, one_row)
    total = -one_row.sum()
    M = sp.vstack([M, one_row], format='csr')
    one_col = np.append(one_row, total)
    M = sp.hstack([M, one_col.reshape(-1, 1)], format='csr')
    return M


def pcg(A, b, epsilon, maxitr, G, Gt):
    n = b.shape[0]
    x = np.zeros(n, dtype=np.double)
    r = b - A * x
    prev_val = 0
    niters = 0
   
    while norm(r) > norm(b) * epsilon and niters < maxitr:
        temp = spsolve_triangular(G, r)
        spsolve_triangular(Gt, temp, lower=False, overwrite_b=True)

        if niters == 0:
            p = temp
        else:
            p = temp + np.dot(r, temp) / prev_val * p

        q = A * p
        alpha = np.dot(p, r) / np.dot(p, q)
        x = x + alpha * p
        prev_val = np.dot(r, temp)
        r = r - alpha * q
        niters = niters + 1
        #print('current residual: ' + str(np.linalg.norm(r) / np.linalg.norm(b)))

    relres = norm(A*x-b)/norm(b)

    #print('# CG iterations: {}'.format(niters))
    #print('Relative residual: {:.2e}'.format(relres))

    return (x, relres, niters)



