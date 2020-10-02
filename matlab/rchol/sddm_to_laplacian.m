% convert given sddm matrix to Laplacian matrix
function [test] = sddm_to_laplacian(original)
    test = original;
    n = size(test, 1);
    condition = sum(test, 2);
    aa = find(abs(condition) < 1e-9);
    %aa = find(condition < 0);
    condition(aa) = 0;
    test = [test -condition];
    test = [test; [(-condition)' sum(condition)]];
end
