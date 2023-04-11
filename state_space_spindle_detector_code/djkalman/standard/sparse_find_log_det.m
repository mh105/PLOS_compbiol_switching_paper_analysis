function log_det = sparse_find_log_det(A)
%% Written by Camilo Lamus (lamus@mit.edu), modified by Alex He
% This uses Schur complement to calculate the log determinant instead of
% using the MATLAB det() function. This is more efficient and more
% numerically stable.

% NOTE: on MATLAB this is actually slower than log(det(A)) regardless of
% sparsity and on CPU or on GPU. But it is more robust by accumulating the
% logs instead of multiplying to get determinant first.

log_det = log(A(1,1));
for i = 1:(size(A,1)-1)
    A = A(2:end,2:end)-A(2:end,1)*A(1,2:end)/A(1,1);
    log_det = log_det + log(A(1,1));
end
end