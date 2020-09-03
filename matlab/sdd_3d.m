% generate 3d sdd matrix
function A = sdd_3d(n)
  e = ones(n, 1);
  I = speye(n);
  D = spdiags([-e 2*e -e], [-1 0 1], n, n);
  A1 = kron(D, I);
  A2 = kron(I, D);
  A = kron(A1, -I) + kron(I, A1) + kron(I, A2) + 4*speye(n*n*n);
end
