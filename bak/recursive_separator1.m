% calculate the separator indices
function [p, val, separator] = recursive_separator1(logic, depth, target)
    if (depth == target)
        size = length(logic);
        val = size;
        p = amd(logic);
        separator = [];
    elseif (length(logic) <= 1)
        size = length(logic);
        [p1, v1] = recursive_separator1([], depth + 1, target);
        [p2, v2] = recursive_separator1(zeros(size, size), depth + 1, target);
        val = [v1 v2 0];
        p = [p1, p2];
        separator = [];
    else
        
        sep = trygraph(logic);
        if depth == 1
            sep(end) = 2;
        end
        
        l = find(sep==0);
        r = find(sep==1);
        s = find(sep==2);
        
        newleft = logic(l, l);
        newright = logic(r, r);
        

        [p1, v1, s1] = recursive_separator1(newleft, depth + 1, target);
        [p2, v2, s2] = recursive_separator1(newright, depth + 1, target);
        separator = [l(s1), r(s2), s];
        
        val = [v1 v2 length(s)];
        p = [l(p1), r(p2), s];
        
    end
    
end