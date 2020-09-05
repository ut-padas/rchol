% convert given sddm matrix to singular matrix
function [test] = convert2singular(original)
    test = original;
    n = size(test, 1);
    condition = sum(test, 2);
    aa = find(abs(condition) < 1e-9);
    %aa = find(condition < 0);
    condition(aa) = 0;
    test = [test -condition];
    test = [test; [(-condition)' sum(condition)]];
end