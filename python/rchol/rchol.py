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

