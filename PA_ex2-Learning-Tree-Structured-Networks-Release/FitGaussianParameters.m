function [mu sigma] = FitGaussianParameters(X)
% X: (N x 1): N examples (1 dimensional)
% Fit N(mu, sigma^2) to the empirical distribution
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

mu = 0;
sigma = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE

%MEAN
% number of samples
nSamples = size(X,1);

% sum up features X_i (cols)
sum_X = sum(X,1);
%
mu = sum_X./nSamples;

%STD
X_squared = X.^2;
sum_X_squared = sum(X_squared,1);
%
mu_squared = sum_X_squared./nSamples;

var = mu_squared - mu.^2;
sigma = sqrt(var);

%%%%%%%%%%%%%%%%%%%%%%%%%%