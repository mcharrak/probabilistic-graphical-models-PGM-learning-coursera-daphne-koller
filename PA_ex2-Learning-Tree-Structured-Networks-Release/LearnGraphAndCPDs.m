function [P G loglikelihood] = LearnGraphAndCPDs(dataset, labels)

% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha) 
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);
% B... # of body parts
B = size(dataset,2);

G = zeros(10,2,K); % graph structures to learn
% initialization
for k=1:K
    G(2:end,:,k) = ones(9,2);
end

% estimate graph structure for each class
for k=1:K
    % fill in G(:,:,k)
    % use ConvertAtoG to convert a maximum spanning tree to a graph G
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    idxs_classK = boolean(labels(:,k));
    dataset_classK = dataset(idxs_classK,:,:);
    % second output is weight matrix W_k (not needed here)
    [A_k, ~] = LearnGraphStructure(dataset_classK);
    G(:,:,k) = ConvertAtoG(A_k);
    %%%%%%%%%%%%%%%%%%%%%%%%%
end

% estimate parameters

% P.c = zeros(1,K);
% compute P.c
% the following code can be copied from LearnCPDsGivenGraph.m
% with little or no modification
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
loglikelihood = 0;
P.c = zeros(1,K);
P_clg_fields = {'mu_y', 'sigma_y', 'mu_x', 'sigma_x', 'mu_angle', 'sigma_angle', 'theta'};
cells = cell(length(P_clg_fields),10);
P.clg = cell2struct(cells,P_clg_fields);

% estimate parameters
% fill in P.c, MLE for class probabilities
% fill in P.clg for each body part and each class
% choose the right parameterization based on G(i,1)
% compute the likelihood - you may want to use ComputeLogLikelihood.m
% you just implemented.
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE

% 1. determine if shared graph over all classes or individual graph for each
% class
if length(size(G)) == 2
    % replicate G for all K classes
    G = repmat(G,[1,1,K]);
end

% 2. calculate P.c using MLE
% count of each class
count_c1 = sum(labels(:,1));
count_c2 = sum(labels(:,2));
P.c(1) = count_c1/N;
P.c(2) = count_c2/N;
% 3. calculate P.clg
% loop over each bodypart
for i = 1:B
    % loop over each class k
    for k = 1:K
        class_k_idxs = find(labels(:,k));
        % case that node i has no parent other than class label
        if G(i,1,k) == 0
            [mu_y, sigma_y] = FitGaussianParameters(dataset(class_k_idxs,i,1));
            [mu_x, sigma_x] = FitGaussianParameters(dataset(class_k_idxs,i,2));
            [mu_angle, sigma_angle] = FitGaussianParameters(dataset(class_k_idxs,i,3));
            % populate mus
            P.clg(i).mu_y(1,k) = mu_y;
            P.clg(i).mu_x(1,k) = mu_x;
            P.clg(i).mu_angle(1,k) = mu_angle;
            % populate sigmas
            P.clg(i).sigma_y(1,k) = sigma_y;
            P.clg(i).sigma_x(1,k) = sigma_x;
            P.clg(i).sigma_angle(1,k) = sigma_angle;  
            % populate theta
            % leave theta blank/empty
        % case that node i has additional parent      
        else
            % determine idx of parent for node i
            p_i = G(i,2,k);
            data_parent = dataset(class_k_idxs,p_i,:);
            % data_parent has size Nx1x3; need to remove dim of len 1
            data_parent = squeeze(data_parent);
            % y
            data_y = dataset(class_k_idxs,i,1);
            [Beta_y, sigma_y] = FitLinearGaussianParameters(data_y,data_parent);
            % x
            data_x = dataset(class_k_idxs,i,2);
            [Beta_x, sigma_x] = FitLinearGaussianParameters(data_x,data_parent);
            % angle
            data_angle = dataset(class_k_idxs,i,3);
            [Beta_angle, sigma_angle] = FitLinearGaussianParameters(data_angle,data_parent);                                                      
            % populate sigmas
            P.clg(i).sigma_y(1,k) = sigma_y;
            P.clg(i).sigma_x(1,k) = sigma_x;
            P.clg(i).sigma_angle(1,k) = sigma_angle;
            % populate theta (which has shape: 2x12)
            % last beta goes to first position of theta values
            % -> refer to equations (7) & (8) from assignment sheet
            theta_vec = [Beta_y(4),Beta_y(1:3)',...
                         Beta_x(4), Beta_x(1:3)',...
                         Beta_angle(4), Beta_angle(1:3)'];            
            P.clg(i).theta(k,:) = theta_vec;
        end
    end
end    
% These are dummy lines added so that submit.m will run even if you 
% have not started coding. Please delete them.
% P.clg.sigma_x = 0;
% P.clg.sigma_y = 0;
% P.clg.sigma_angle = 0;


% 4. calculate loglikelihood with P
loglikelihood = loglikelihood + ComputeLogLikelihood(P, G, dataset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('log likelihood: %f\n', loglikelihood);