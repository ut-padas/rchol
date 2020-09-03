% convert sdd matrix into sddm matrix
function [sddm] = sdd_to_sddm(original)
    % diagonal
    di = diag(diag(original));
    
    % find positive and negative spots
    nodiag = original - di;
    posidx = find(nodiag > 0);
    negidx = find(nodiag < 0);
    
    % N and P matrices
    pos = nodiag;
    neg = nodiag;
    pos(negidx) = 0;
    neg(posidx) = 0;
    
    sddm = [di + neg, -pos; 
     -pos, di + neg];
end
