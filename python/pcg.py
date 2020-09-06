import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import spsolve_triangular

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

