addpath('rchol/')

% 3D constant Poisson equation with dirichlet boundary
n = 64;
A = laplace_3d(n); % see 'rchol/laplace_3d.m'

% random RHS
b = rand(size(A, 1), 1);

% compute preconditioner after reordering
p = amd(A);
G = rchol(A(p, p));
fprintf('fill-in ratio: %.2e\n', 2*nnz(G)/nnz(A))

% solve with PCG
tol = 1e-6;
maxit = 200;
[x, flag, relres, itr] = pcg(A(p,p), b(p), tol, maxit, G, G');
fprintf('# CG iterations: %d\n', itr)

% verify solution
y = zeros(length(x), 1);
y(p) = x;
fprintf('Relative residual: %.2e\n', norm(b-A*y)/norm(b))

