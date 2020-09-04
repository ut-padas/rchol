% 3D SDD matrix from hyperbolic equation
n = 16;
A = hyperbolic_3d(n);
%A = hyperbolic_2d(n);

% random RHS
b = rand(size(A, 1), 1);

% create extended matrix
Ae = sdd_to_sddm(A);
be = [b; -b];

% compute preconditioner after reordering
p = amd(Ae);
G = rchol(Ae(p,p));

% solve with PCG
tol = 1e-6;
maxit = 100;
[x, flag, relres, itr] = pcg(Ae(p,p), be(p), tol, maxit, G, G');
fprintf('\n')
fprintf('flag: %d\n', flag)
fprintf('# iterations: %d\n', itr)
fprintf('relative residual: %.2e\n', relres)

% verify solution
y = zeros(length(x), 1);
y(p) = x;
fprintf('fill ratio: %.2e\n', 2*nnz(G)/nnz(Ae))
fprintf('Verify residual: %.2e\n', norm(b-A*y(1:length(b)))/norm(b))


Gi = ichol(A, struct('type','ict','droptol',1e-1));
[x, flag, relres, itr] = pcg(A, b, tol, maxit, Gi, Gi');
fprintf('\n')
fprintf('flag: %d\n', flag)
fprintf('# iterations: %d\n', itr)
fprintf('fill ratio: %.2e\n', 2*nnz(Gi)/nnz(A))
fprintf('relative residual: %.2e\n', relres)



