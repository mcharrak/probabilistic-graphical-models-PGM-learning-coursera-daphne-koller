function [Beta sigma] = FitLinearGaussianParameters(X, U)

% Estimate parameters of the linear Gaussian model:
% X|U ~ N(Beta(1)*U(1) + ... + Beta(n)*U(n) + Beta(n+1), sigma^2);

% Note that Matlab/Octave index from 1, we can't write Beta(0).
% So Beta(n+1) is essentially Beta(0) in the text book.

% X: (M x 1), the child variable, M examples
% U: (M x N), N parent variables, M examples
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

% M... number of examples/samples
M = size(U,1);
% N... number of (for our tree always N=2 (parent nodes: O and C)
N = size(U,2);

Beta = zeros(N+1,1);
sigma = 1;

% collect expectations and solve the linear system
% A = [ E[U(1)],      E[U(2)],      ... , E[U(n)],      1     ; 
%       E[U(1)*U(1)], E[U(2)*U(1)], ... , E[U(n)*U(1)], E[U(1)];
%       ...         , ...         , ... , ...         , ...   ;
%       E[U(1)*U(n)], E[U(2)*U(n)], ... , E[U(n)*U(n)], E[U(n)] ]

% construct A
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE

% matrix we need to fill up
A = ones(N+1,N+1);
% last column of A
mean_U = mean(U);
last_col = [1;mean_U'];
% first row of A 
first_row = [mean_U,1];
% convert A of dim NxN to final A of dim (N+1)x(N+1)
% stack row as top row
A(1,:) = first_row;
% attach column as final col
A(:,end) = last_col;
% loop with elementwise vector multiplications (hadamard products) followed
% by expectation operator (simple mean)
for i = 1:N
    for j = 1:N
        prod_ij = U(:,i).*U(:,j);
        E_ij = mean(prod_ij);
        % fill in final value
        A(i+1,j) = E_ij;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% B = [ E[X]; E[X*U(1)]; ... ; E[X*U(n)] ]

% construct B
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE

B = ones(N+1,1);
% fill in 1st row of B
first_row_B = mean(X);
B(1,1) = first_row_B;
for k = 1:N
    prod_X_U_k = X.*U(:,k);
    E_X_U_k = mean(prod_X_U_k);
    B(k+1,1) = E_X_U_k;
end    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% solve A*Beta = B
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
% use backslash operator to olve systems of linear equations A*x=b
Beta = A\B;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% then compute sigma according to eq. (11) in PA description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
% use flag w=1 to normalize by number of observations instead of number of
% observations-1
% cov_X is scalar
cov_X_X = mean(X.*X) - mean(X)^2;
% cov_U is matrix
cov_U = cov(U,1);
% multiply Beta values into cov_U
% we need all but last beta value
beta_vec = Beta(1:end-1,:);
% beta_vec is Nx1
beta_mat = beta_vec * beta_vec';
prod_cov_U_beta = cov_U.*beta_mat;

first_term = cov_X_X;
second_term = sum(sum(prod_cov_U_beta));
sigma = sqrt(first_term - second_term);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%