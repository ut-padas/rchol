% 3D constant Poisson equation with dirichlet boundary
n = 64;
A = laplace_3d(n);

% random RHS
b = rand(size(A, 1), 1);

% compute preconditioner after reordering
thread = 2;
[G, p] = rchol(A, thread);
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

