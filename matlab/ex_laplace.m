% 3D constant Poisson equation with dirichlet boundary
n = 8;
A = laplace_3d(n);

% random RHS
b = rand(size(A, 1), 1);

% compute preconditioner after reordering
p = amd(A);
G = rchol(A(p,p));

% solve with PCG
tol = 1e-6;
maxit = 50;
[x, flag, relres] = pcg(A(p,p), b(p), tol, maxit, G, G');

% verify solution
fprintf('Relative residual: %e.2\n', norm(b-A*x(p))/norm(b))



