addpath('rchol/')

% SDD matrix by flipping the '-1' corresponding to z direction in 3D Laplace
n = 16;
A = sdd_3d(n); % see ./rchol/sdd_3d.m

% random RHS
N = size(A, 1);
b = rand(N, 1);

% compute indices for one connected component
nxy = 1:n*n; % indices of a 2D grid
nz = (0:2:n-1)*n*n; % strided indices in z direction
I1 = (nxy'+nz);
I1 = I1(:);
I2 = setdiff(1:N,I1);
I2 = I2(:);

% construct the reduced problem by flipping signs of to positive entries
Ag = A;
Ag(I1,I2) = -A(I1,I2);
Ag(I2,I1) = -A(I2,I1);

bg = b;
bg(I2) = -b(I2);

% solve with PCG
tol = 1e-6;
maxit = 200;
p = amd(Ag);
G = rchol(Ag(p,p));
[xg, flag, relres, itr] = pcg(Ag(p,p), bg(p), tol, maxit, G, G');
fprintf('matrix size: %d x %d\n', size(Ag,1), size(Ag,2))
fprintf('fill ratio: %.2f\n', 2*nnz(G)/nnz(Ag))
fprintf('# iterations: %d\n', itr)

% retrieve original solution
xg(p) = xg;
x = xg;
x(I2) = -xg(I2);

% check residual
fprintf('relative residual: %.2e\n', norm(b - A*x)/norm(b))


