is_sddm = 1;
n = 32;
dim = 5;
%original = Problem.A;
original = laplace_3d(n + 1);
%original = laplace(n, dim);
if is_sddm == 0
    sddm = convert2sddm(original);
    test = convert2singular(sddm);
else
    test = convert2singular(original);
end
%
graphtest = test - diag(diag(test));
logic = (graphtest ~= 0);
index = 1 : length(graphtest);
tic;
[p_vec, val, sep] = recursive_separator1(logic, 1, 1);
toc;

%p_vec = amd(test);
%p_vec = randperm(size(test, 1));

idx = find(p_vec == size(test, 1));
p_vec(idx) = p_vec(end);
p_vec(end) = size(test, 1);


give = tril(test(p_vec, p_vec)) * -1;
separator = zeros(1, size(give, 1));
separator(sep) = 1;
separator = separator(p_vec);
result_idx = [0 cumsum(val)];
save('../128/give.mat', 'give');
save('../128/separator.mat', 'separator');
save('../128/val.mat', 'val');
save('../128/result_idx.mat', 'result_idx');
save('../128/p_vec.mat', 'p_vec');
b = rand(size(original, 1), 1);
%b = b - mean(b);
if is_sddm == 0
    b = [b; -b];
end
c = [b; -sum(b)];
d = c;
c = c(p_vec);
save('../128/rightside.mat', 'c');
c = d;

% testing vectors
%{
L1 = load('../128/lower.mat', '-mat', 'lower');
D1 = load('../128/diag.mat', '-mat', 'diag');
L1 = L1.lower;
D1 = D1.diag;
%}
[D1, L1] = matlab_solve(give, result_idx, 1);

lastrow = nnz(L1(end, :));
L1 = L1(1 : end - 1, 1 : end - 1);
D1 = D1(1 : end - 1);

% matlab incomplete
%{
iC = ichol(test(p_vec, p_vec), struct('type', 'ict', 'droptol', 0.07e-2));
density_iC = 2*nnz(iC)/nnz(test)
x = pcg(test(p_vec, p_vec),c(p_vec, :),1e-10,1000,iC,iC',[]);
P = speye(size(test, 1));
P = P(p_vec, :)';
x = P * x;
x = x(1 : size(b, 1)) - x(size(b, 1) + 1) * ones(size(b, 1), 1);
if is_sddm == 0
    disp("matlab incomplete: " + norm(original*x(1 : length(x) / 2)-b(1 : length(x) / 2)) / norm(b(1 : length(x) / 2)));
else
    disp("matlab incomplete: " + norm(original*x-b) / norm(b));
end


% matlab incomplete original permutation
%0.005e-2
gg = tic;
iC = ichol(original, struct('type', 'ict', 'droptol', 0.19e-2));
icholtime = toc(gg)
density_iC = 2*nnz(iC)/nnz(original)
if is_sddm == 0
    [x,flag,relres,iter,resvec] = pcg(original, b(1 : length(b) / 2), 1e-10,1000,iC,iC',[]);
else
    s1 = tic;
    %[x,flag,relres,iter,resvec] = pcg(original, b, 1e-10,1000,iC,iC',[]);
    x = pcg(original, b, 1e-13,1000,iC,iC',[]);
    incomp_pcgtime = toc(s1)
end
if is_sddm == 0
    disp("matlab incomplete(original): " + norm(original*x-b(1 : length(b) / 2)) / norm(b(1 : length(b) / 2)));
else
    disp("matlab incomplete(original): " + norm(original*x-b) / norm(b));
end
%}

% Julia sampling
func = @(x) cSolver(L1, D1, x);
density_julia = 2*nnz(L1)/nnz(original)
%[y,flag1,relres1,iter1,resvec1] = pcg(original(p_vec(1 : end - 1), p_vec(1 : end - 1)), b(p_vec(1 : end - 1), :),1e-13,200,func,[]);
s1 = tic;
y = pcg(original(p_vec(1 : end - 1), p_vec(1 : end - 1)), b(p_vec(1 : end - 1), :), 1e-13,200,func,[]);
pcgtime = toc(s1)
%j = y;
%P = speye(size(test, 1));
%P = P(p_vec, :)';
%j = P * j;
%j = j(1 : size(b, 1)) - j(size(b, 1) + 1) * ones(size(b, 1), 1);
pt = 1 : length(p_vec) - 1;
pt(p_vec(1 : end - 1)) = 1 : length(p_vec) - 1;
y = y(pt);
if is_sddm == 0
    disp("random sampling: " + norm(original*y(1 : length(b) / 2)-b(1 : length(b) / 2)) / norm(b(1 : length(b) / 2)));
else
    disp("random sampling: " + norm(original*y-b) / norm(b));
end

figure;
spy(L1(1 : 100000, 1 : 100000));
%set(gca,'FontSize',30)
%set(gca,'XTick',[], 'YTick', [])
set(gca,'xticklabel',{[]})
set(gca,'yticklabel',{[]})
delete(findall(findall(gcf,'Type','axe'),'Type','text'))

% solver function used for pcg
function [y] = cSolver(L, D, y)
    n = size(L, 1);
    y = L\y;
    y = y./D;
    y = L'\y;
 
end

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

function A = laplace_3d(n)
  N = (n - 1)^3;  % total number of grid points

  % set up sparse matrix
  idx = zeros(n+1,n+1,n+1);  % index mapping to each point, including "ghosts"

  idx(2:n,2:n,2:n) = reshape(1:N,n-1,n-1,n-1);

  mid = 2:n;    % "middle" indices -- interaction with self

  lft = 1:n-1;  % "left"   indices -- interaction with one below

  rgt = 3:n+1;  % "right"  indices -- interaction with one above

  I = idx(mid,mid,mid); e = ones(size(I));

  % interactions with ...

  Jl = idx(lft,mid,mid); Sl = -e;                              % left

  Jr = idx(rgt,mid,mid); Sr = -e;                              % right

  Ju = idx(mid,lft,mid); Su = -e;                              % up

  Jd = idx(mid,rgt,mid); Sd = -e;                              % down

  Jf = idx(mid,mid,lft); Sf = -e;                              % front

  Jb = idx(mid,mid,rgt); Sb = -e;                              % back

  Jm = idx(mid,mid,mid); Sm = -(Sl + Sr + Sd + Su + Sb + Sf);  % middle (self)

  % combine all interactions

  I = [ I(:);  I(:);  I(:);  I(:);  I(:);  I(:);  I(:)];

  J = [Jl(:); Jr(:); Ju(:); Jd(:); Jf(:); Jb(:); Jm(:)];

  S = [Sl(:); Sr(:); Su(:); Sd(:); Sf(:); Sb(:); Sm(:)];

  % remove ghost interactions

  idx = find(J > 0); I = I(idx); J = J(idx); S = S(idx);

  A = sparse(I,J,S,N,N);
  %A = A - spdiags(sum(A, 2), 0, size(A, 1), size(A, 2));
    
end


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

function [sddm] = convert2sddm(original)
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