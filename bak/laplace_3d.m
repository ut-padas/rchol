
% generate 3d laplace problem
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
  save('I.mat', 'I', '-v7.3');  
  save('J.mat', 'J', '-v7.3');
  save('S.mat', 'S', '-v7.3');
end
