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

% verify solution
y = zeros(length(x), 1);
y(p) = x;
fprintf('Relative residual: %.2e\n', norm(b-A*y)/norm(b))



