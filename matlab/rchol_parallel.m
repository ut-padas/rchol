function [G, p] = rchol_parallel(A, thread)
    n = size(A,1);
    depth = floor(log2(thread)) + 1;
    [p, val, ~] = find_separator(A - diag(diag(A)), 1, depth);
    result_idx = [0 cumsum(val)];
    result_idx(end) = result_idx(end) + 1;
    Lap = sddm_to_laplacian(A(p, p));
    edges = -1 * tril(Lap);
    [D, L] = rchol_lap(edges, result_idx, 2^(depth - 1));
    G = L(1:n,1:n)*spdiags(sqrt(D(1:n)), 0, n, n);
end
