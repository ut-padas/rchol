function G = rchol(A)
    n = size(A,1);
    L = sddm_to_laplacian(A);
    edges = -1 * tril(L);
    nt = 1;
    [D, L] = matlab_solve(edges, [0, n+1], nt);
    G = L(1:n,1:n).*D(1:n);
end