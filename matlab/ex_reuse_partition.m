addpath('rchol/')

% initial 3D-Poisson matrix
n = 64;
A = laplace_3d(n); % see ./rchol/laplace_3d.m 

% random RHS
b = rand(size(A, 1), 1);
fprintf('Initial problem ...\n')

% compute preconditioner and the associated permutation/partition
thread = 2;
[G, perm, part] = rchol(A, thread);
fprintf('fill-in ratio: %.2e\n', 2*nnz(G)/nnz(A))

% solve the initial problem
tol = 1e-6;
maxit = 200;
[x, flag, relres, itr] = pcg(A(perm,perm), b(perm), tol, maxit, G, G');
fprintf('# CG iterations: %d\n', itr)
fprintf('Relative residual: %.2e\n\n', relres)


% perturb the initial matrix
B = A + 1e-3*speye(size(A,1));
fprintf('New problem (same sparsity) ...\n')


% compute preconditioner with existing permutation/partition
L = rchol(B, thread, perm, part);
fprintf('fill-in ratio: %.2e\n', 2*nnz(L)/nnz(B))

% solve the new problem
[x, flag, relres, itr] = pcg(B(perm,perm), b(perm), tol, maxit, L, L');
fprintf('# CG iterations: %d\n', itr)
fprintf('Relative residual: %.2e\n', relres)




