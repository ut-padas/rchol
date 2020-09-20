% calculate the separator indices
function [p, val, separator] = find_separator(logic, depth, target)
    if (depth == target)
        size = length(logic);
        val = size;
        p = amd(logic);
        separator = [];
    elseif (length(logic) <= 1)
        size = length(logic);
        [p1, v1] = find_separator([], depth + 1, target);
        [p2, v2] = find_separator(zeros(size, size), depth + 1, target);
        val = [v1 v2 0];
        p = [p1, p2];
        separator = [];
    else
        
        sep = metis_separator(logic);
        
        l = find(sep==0);
        r = find(sep==1);
        s = find(sep==2);
        
        newleft = logic(l, l);
        newright = logic(r, r);
        

        [p1, v1, s1] = find_separator(newleft, depth + 1, target);
        [p2, v2, s2] = find_separator(newright, depth + 1, target);
        separator = [l(s1), r(s2), s];
        
        val = [v1 v2 length(s)];
        p = [l(p1), r(p2), s];
        
    end
    
end