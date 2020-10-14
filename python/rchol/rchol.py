import numpy as np
from scipy.sparse import triu, diags
from pyrchol_lap import *
from find_separator import *
from util import *
import math

def rchol(A, nt=1, perm=np.array([]), part=np.array([])):

  # check parameter
  if nt < 1:
    raise Exception("number of threads must be at least 1")
  if (perm.size>0) ^ (part.size>0):
    raise Exception("permutation and partition must be given together")

  # only power of 2 is supported
  if not math.log2(nt).is_integer():
    nt = math.floor(math.log2(threads))
    print('number of threads rounded down to {}'.format(nt))
  
  # get matrix size
  n = A.shape[0]

  # compute permutation and partition
  if perm.size == 0:
    if nt == 1:
      perm = np.arange(n)
      part = np.array([0, n+1], dtype=np.uint64)
    else:
      depth = math.log2(nt) + 1
      Diag = diags(A.diagonal(), format='csr')
      perm, val, _ = find_separator(A - Diag, 1, depth)
      part = np.append(0, np.cumsum(val))
      part[-1] = part[-1] + 1

  # factorization
  Lap = sddm_to_laplacian(A[perm[:,None], perm])
  edges = -1*triu(Lap, format='csr')

  L, D = rchol_lap(edges, edges.shape[0] - 1, nt, part)
  L = L.transpose()
  Gt = L * diags(np.sqrt(D))


  return Gt.tocsr(), perm, part


