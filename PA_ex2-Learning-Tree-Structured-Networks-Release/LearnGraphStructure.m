function [A W] = LearnGraphStructure(dataset)

% Input:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% 
% Output:
% A: maximum spanning tree computed from the weight matrix W
% W: 10 x 10 weight matrix, where W(i,j) is the mutual information between
%    node i and j. 
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset,1); % number of samples in dataset
K = size(dataset,3); % number of values for a single observation (y,x,angle)
B = size(dataset,2); % number of body parts

W = zeros(10,10);
% Compute weight matrix W
% set the weights following Eq. (14) in PA description
% you don't have to include M since all entries are scaled by the same M
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE        
for i = 1:B
    Oi = squeeze(dataset(:,i,:));
    for j = i:B
        Oj = squeeze(dataset(:,j,:));
        MI = GaussianMutualInformation(Oi,Oj);
        W(i,j) = MI;
    end
end

add_tran_W = W + W';
off_diagonal_W = eye(B).* W;
clear W;
W = add_tran_W - off_diagonal_W;
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute maximum spanning tree
A = MaxSpanningTree(W);