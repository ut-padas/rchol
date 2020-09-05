% decide whether the problem is sddm and generate problem
is_sddm = 1;
n = 10;
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

    
% calculate separator indices
graphtest = test - diag(diag(test));
logic = (graphtest ~= 0);
index = 1 : length(graphtest);
tic;
[p_vec, val, ~] = recursive_separator1(logic, 1, 2);
toc;
idx = find(p_vec == size(test, 1));
p_vec(idx) = p_vec(end);
p_vec(end) = size(test, 1);
give = tril(test(p_vec, p_vec)) * -1;
result_idx = [0 cumsum(val)];



% testing right side vectors
b = rand(size(original, 1), 1);
if is_sddm == 0
    b = [b; -b];
end


% calculate the preconditioner, convert it back to sddm
[D1, L1] = matlab_solve(give, result_idx, 1);
lastrow = nnz(L1(end, :));
L1 = L1(1 : end - 1, 1 : end - 1);
D1 = D1(1 : end - 1);




% solve using pcg
func = @(x) cSolver(L1, D1, x);
density_julia = 2*nnz(L1)/nnz(original)
s1 = tic;
%[y,flag1,relres1,iter1,resvec1] = pcg(original(p_vec(1 : end - 1), p_vec(1 : end - 1)), b(p_vec(1 : end - 1), :),1e-13,200,func,[]);
y = pcg(original(p_vec(1 : end - 1), p_vec(1 : end - 1)), b(p_vec(1 : end - 1), :), 1e-13,200,func,[]);
pcgtime = toc(s1)
pt = 1 : length(p_vec) - 1;
pt(p_vec(1 : end - 1)) = 1 : length(p_vec) - 1;
y = y(pt);
if is_sddm == 0
    disp("random sampling: " + norm(original*y(1 : length(b) / 2)-b(1 : length(b) / 2)) / norm(b(1 : length(b) / 2)));
else
    disp("random sampling: " + norm(original*y-b) / norm(b));
end








