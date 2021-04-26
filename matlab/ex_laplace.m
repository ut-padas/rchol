addpath('rchol/')

% 3D constant Poisson equation with dirichlet boundary
n = 128;
A = laplace_3d(n); % see 'rchol/laplace_3d.m'
fprintf('N: %d\n', size(A,1))

% random RHS
b = rand(size(A, 1), 1);

% compute preconditioner after reordering
tic
p = amd(A);
%p = 1:size(A,1);
G = rchol(A(p, p));
t1 = toc;
fprintf('fill-in ratio: %.2e\n', 2*nnz(G)/nnz(A))

% solve with PCG
tol = 1e-10;
maxit = 200;

tic
[x, flag, relres, itr] = pcg(A(p,p), b(p), tol, maxit, G, G');
t2 = toc;

fprintf('# CG iterations: %d\n', itr)

% verify solution
y = zeros(length(x), 1);
y(p) = x;
fprintf('Relative residual: %.2e\n', norm(b-A*y)/norm(b))
fprintf('Setup time: %.2f\n', t1)
fprintf('PCG time: %.2f\n', t2)


