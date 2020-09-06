import numpy as np
from scipy.sparse import triu, diags
import python_interface as pyi
from sddm_to_laplacian import *

def rchol(A):
  Lap = sddm_to_laplacian(A)
  edges = -1*triu(Lap, format='csr')
  L, D = pyi.rchol_lap(edges)
  Gt = L * diags(np.sqrt(D))
  return Gt.transpose().tocsr()

