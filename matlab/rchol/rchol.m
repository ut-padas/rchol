function [G, p] = rchol(A, varargin)

    pa = inputParser;
    addRequired(pa, 'sparse', @isnumeric);
    addOptional(pa, 'thread', 1, @isscalar);
    parse(pa, A, varargin{:})

    if pa.Results.thread <= 0 || log2(pa.Results.thread) ~= floor(log2(pa.Results.thread))
        ME = MException('Variable thread must be positive and powers of 2');
        throw(ME);
    end
    
    
    thread = pa.Results.thread;
    n = size(A,1);
    if thread == 1
        Lap = sddm_to_laplacian(A);
        edges = -1 * tril(Lap);
        nt = 1;
        [D, L] = rchol_lap(edges, [0, n+1], nt);
        G = L(1:n,1:n)*spdiags(sqrt(D(1:n)), 0, n, n);
        p = 1 : n;
    else
        depth = log2(thread) + 1;
        [p, val, ~] = find_separator(A - diag(diag(A)), 1, depth);
        result_idx = [0 cumsum(val)];
        result_idx(end) = result_idx(end) + 1;
        Lap = sddm_to_laplacian(A(p, p));
        edges = -1 * tril(Lap);
        [D, L] = rchol_lap(edges, result_idx, 2^(depth - 1));
        G = L(1:n,1:n)*spdiags(sqrt(D(1:n)), 0, n, n);
    end
    
end
