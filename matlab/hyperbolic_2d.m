% generate 2d sdd matrix
function A = hyperbolic_2d(n)
  e = ones(n, 1);
  I = speye(n);
  D = spdiags([-e 2*e -e], [-1 0 1], n, n);
  A = kron(D, -I) + kron(I, D) + 4*speye(n*n);
end
