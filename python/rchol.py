import numpy as np
import scipy
from scipy.sparse import triu
import python_interface as pyi

def rchol(A):
  Lap = pyi.sddm_to_laplacian(A)
  edges = -1*triu(Lap, format='csr')
  L, D = pyi.python_factorization(edges)
  Gt = L * scipy.sparse.diags(np.sqrt(D))
  return Gt.transpose().tocsr()

