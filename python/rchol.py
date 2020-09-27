import numpy as np
from scipy.sparse import triu, diags
from pyrchol_lap import *
from util import *
import math


def rchol(A):
  Lap = sddm_to_laplacian(A)
  edges = -1*triu(Lap, format='csr')
  L, D = rchol_lap(edges)
  Gt = L * diags(np.sqrt(D))
  return Gt.transpose().tocsr()




def rchol(A, threads):
  if threads < 1 or not math.log2(threads).is_integer():
    raise Exception("input thread non positive or not powers of 2") 
  
  remove = diags(A.diagonal(), format='csr')
  p, val, _ = find_separator(A - remove, 1, math.log2(threads) + 1)
  result_idx = np.append(0, np.cumsum(val))
  result_idx[-1] = result_idx[-1] + 1

  Lap = sddm_to_laplacian(A[p[:,None], p])
  edges = -1*triu(Lap, format='csr')
  L, D = rchol_lap_cpp(edges, edges.shape[0] - 1, threads, result_idx)
  Gt = L * diags(np.sqrt(D))
  return p, Gt.transpose().tocsr()