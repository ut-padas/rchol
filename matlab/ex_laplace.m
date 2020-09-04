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
[x, flag, relres, itr] = pcg(A(p,p), b(p), tol, maxit, G, G');
fprintf('\n')
fprintf('flag: %d\n', flag)
fprintf('# iterations: %d\n', itr)
fprintf('relative residual: %.2e\n', relres)

% verify solution
y = zeros(length(x), 1);
y(p) = x;
fprintf('fill ratio: %.2e\n', 2*nnz(G)/nnz(A))
fprintf('Relative residual: %.2e\n', norm(b-A*y)/norm(b))


Gi = ichol(A, struct('type','ict','droptol',1e-1));
[x, flag, relres, itr] = pcg(A, b, tol, maxit, Gi, Gi');
fprintf('\n')
fprintf('flag: %d\n', flag)
fprintf('# iterations: %d\n', itr)
fprintf('fill ratio: %.2e\n', 2*nnz(Gi)/nnz(A))
fprintf('relative residual: %.2e\n', relres)

