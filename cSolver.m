% solver function used for pcg
function [y] = cSolver(L, D, y)
    n = size(L, 1);
    y = L\y;
    y = y./D;
    y = L'\y;
end

