function [G, perm, part] = rchol(A, nt, perm, part)

    % default parameter
    if nargin < 2 || isempty(nt), nt = 1; end
    if nargin < 3 || isempty(perm), perm = []; end
    if nargin < 4 || isempty(part), part = []; end
    
    % check parameter
    if nt <= 0 
        ME = MException('number of threads must be positive');
        throw(ME);
    end
    if xor(isempty(perm), isempty(part))
        ME = MException('permutation and partition must be given together');
        throw(ME);
    end
    
    % only power of 2 is supported
    if log2(nt) ~= floor(log2(nt))
        nt = 2^floor(log2(nt));
        warning('number of threads rounded down to %d\n', nt)
    end
   
    % get matrix size
    n = size(A,1);

    if isempty(perm)
        if nt == 1
            perm = 1:n; % no reordering
            part = [0, n+1]; % no partition
        else
            % compute partition
            depth = log2(nt) + 1;
            [perm, val, ~] = find_separator(A - diag(diag(A)), 1, depth);
            part = [0 cumsum(val)];
            part(end) = part(end) + 1;
        end
    end

    % compute factorization
    Lap = sddm_to_laplacian(A(perm, perm));
    edges = -1 * tril(Lap);
    [D, L] = rchol_lap(edges, part, nt);
   

    % compute (approximate) Cholesky factor
    G = L(1:n,1:n)*spdiags(sqrt(D(1:n)), 0, n, n);
end
