function G = rchol(A)
    n = size(A,1);
    Lap = sddm_to_laplacian(A);
    edges = -1 * tril(Lap);
    nt = 1;
    [D, L] = rchol_lap(edges, [0, n+1], nt);
    G = L(1:n,1:n)*spdiags(sqrt(D(1:n)), 0, n, n);
end
