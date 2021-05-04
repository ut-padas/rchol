addpath('rchol/')
addpath('/home1/06108/chaochen/matrices')

maxNumCompThreads(1);
fprintf('Number of threads: %d\n', maxNumCompThreads)

% SDD matrix by flipping the '-1' corresponding to z direction in 3D Laplace
%n = 16;
%A = sdd_3d(n); % see ./rchol/sdd_3d.m
load('offshore')
A = Problem.A;

% random RHS
N = size(A, 1);
b = rand(N, 1);

% create extended matrix
Ae = sdd_to_sddm(A);
be = [b; -b];

% compute preconditioner after reordering
tic
p = amd(Ae);
G = rchol(Ae(p,p));
tf = toc;

% solve with PCG
tol = 1e-10;
maxit = 200;

tic
[xe, flag, relres, itr] = pcg(Ae(p,p), be(p), tol, maxit, G, G');
ts = toc;


fprintf('matrix size: %d x %d\n', size(Ae,1), size(Ae,2))
fprintf('fill ratio: %.2f\n', 2*nnz(G)/nnz(A))
fprintf('# iterations: %d\n', itr)
fprintf('Relative residual: %.2e\n', relres)

% check solution
xe(p) = xe;
x = (xe(1:N)-xe(N+1:end))/2;
fprintf('Verify residual: %.2e\n', norm(b-A*x)/norm(b))
fprintf('Setup time: %.2f\n', tf)
fprintf('Solve time: %.2f\n', ts)

