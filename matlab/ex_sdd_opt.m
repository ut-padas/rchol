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

% extract the connected component
Ae = sdd_to_sddm(A);
be = [b; -b];
I = [I1; I2+N];
Ag = Ae(I, I);
bg = be(I);

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
S = length(I1);
x = zeros(N, 1);
xg(p) = xg;
x(I1) = xg(1:S);
x(I2) = -xg(S+1:end);

% check residual
fprintf('relative residual: %.2e\n', norm(b - A*x)/norm(b))


