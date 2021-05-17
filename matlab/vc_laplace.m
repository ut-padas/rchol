% This script is adopted from https://github.com/klho/FLAM/blob/master/hifde/test/fd_cube2.m
function A = vc_laplace(n, rho);
  % initialize
  n = n + 1;
  N = (n - 1)^3;  % total number of grid points

  % set up conductivity field
  a = zeros(n+1,n+1,n+1);
  A = rand(n-1,n-1,n-1);  % random field
  A = fftn(A,[2*n-3 2*n-3 2*n-3]);
  [X,Y,Z] = ndgrid(0:n-2);
  % Gaussian smoothing over 4 grid points
  C = gausspdf(X,0,4).*gausspdf(Y,0,4).*gausspdf(Z,0,4);
  B = zeros(2*n-3,2*n-3,2*n-3);
  B(1:n-1,1:n-1,1:n-1) = C;
  B(1:n-1,1:n-1,n:end) = C( :   , :   ,2:n-1);
  B(1:n-1,n:end,1:n-1) = C( :   ,2:n-1, :   );
  B(1:n-1,n:end,n:end) = C( :   ,2:n-1,2:n-1);
  B(n:end,1:n-1,1:n-1) = C(2:n-1, :   , :   );
  B(n:end,1:n-1,n:end) = C(2:n-1, :   ,2:n-1);
  B(n:end,n:end,1:n-1) = C(2:n-1,2:n-1, :   );
  B(n:end,n:end,n:end) = C(2:n-1,2:n-1,2:n-1);
  B(:,:,n:end) = flipdim(B(:,:,n:end),3);
  B(:,n:end,:) = flipdim(B(:,n:end,:),2);
  B(n:end,:,:) = flipdim(B(n:end,:,:),1);
  B = fftn(B);
  A = ifftn(A.*B);        % convolution in Fourier domain
  A = A(1:n-1,1:n-1,1:n-1);
  idx = A > median(A(:));
  A( idx) = sqrt(rho);         % set upper 50% to something large
  A(~idx) = 1/sqrt(rho);         % set lower 50% to something small
  a(2:n,2:n,2:n) = A;
  clear X Y Z A B C

  % set up sparse matrix
  idx = zeros(n+1,n+1,n+1);  % index mapping to each point, including "ghosts"
  idx(2:n,2:n,2:n) = reshape(1:N,n-1,n-1,n-1);
  mid = 2:n;    % "middle" indices -- interaction with self
  lft = 1:n-1;  % "left"   indices -- interaction with one below
  rgt = 3:n+1;  % "right"  indices -- interaction with one above
  I = idx(mid,mid,mid);
  % interactions with ...
  Jl = idx(lft,mid,mid); Sl = -0.5*(a(lft,mid,mid) + a(mid,mid,mid));  % left
  Jr = idx(rgt,mid,mid); Sr = -0.5*(a(rgt,mid,mid) + a(mid,mid,mid));  % right
  Ju = idx(mid,lft,mid); Su = -0.5*(a(mid,lft,mid) + a(mid,mid,mid));  % up
  Jd = idx(mid,rgt,mid); Sd = -0.5*(a(mid,rgt,mid) + a(mid,mid,mid));  % down
  Jf = idx(mid,mid,lft); Sf = -0.5*(a(mid,mid,lft) + a(mid,mid,mid));  % front
  Jb = idx(mid,mid,rgt); Sb = -0.5*(a(mid,mid,rgt) + a(mid,mid,mid));  % back
  Jm = idx(mid,mid,mid); Sm = -(Sl + Sr + Sd + Su + Sb + Sf);  % middle (self)
  % combine all interactions
  I = [ I(:);  I(:);  I(:);  I(:);  I(:);  I(:);  I(:)];
  J = [Jl(:); Jr(:); Ju(:); Jd(:); Jf(:); Jb(:); Jm(:)];
  S = [Sl(:); Sr(:); Su(:); Sd(:); Sf(:); Sb(:); Sm(:)];
  % remove ghost interactions
  idx = find(J > 0); I = I(idx); J = J(idx); S = S(idx);
  A = sparse(I,J,S,N,N);
end
