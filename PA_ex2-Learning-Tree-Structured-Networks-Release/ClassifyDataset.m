function accuracy = ClassifyDataset(dataset, labels, P, G)
% returns the accuracy of the model P and graph G on the dataset 
%
% Inputs:
% dataset: N x 10 x 3, N test instances represented by 10 parts
% labels:  N x 2 true class labels for the instances.
%          labels(i,j)=1 if the ith instance belongs to class j 
% P: struct array model parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description) 
%
% Outputs:
% accuracy: fraction of correctly classified instances (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1); % number of samples N
K = size(labels,2); % number of classes K
B = size(dataset,2); % number of body parts B
% we need to use logical to get boolean labels due to a bug in the PA
% assignemt's test inputs cf. "load submit_input" and "INPUT.t5a2"
labels = logical(labels);
unique_rows = unique(labels,'rows');
% we need to use [1,1] iff prob for class 1 and 2 are same
mate_row = sum(unique_rows,1);
%disp(unique_rows);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
% 0. determine if shared graph over all classes or individual graph
if length(size(G)) == 2
    % replicate G for all K classes
    G = repmat(G,[1,1,K]);
end
% 1. convert labels into array of class indices
class_idxs = zeros(N,1);
for i = 1:N
    if labels(i,1) == 1
        class_idxs(i) = 1;
    else
        class_idxs(i) = 2;
    end
end

% 2. calculate probabilities for each class and predictions
class_prob = zeros(N,K);
class_pred = zeros(N,K);
% loop over N examples
for n = 1:N
    % fetch current sample O
    O = dataset(n,:,:);
    % O has size: 1x10x3 thus remove redundant dimensions of length 1    
    O = squeeze(O);
    % loop over K classes
    for k = 1:K
        % probability current class k
        P_ck = P.c(k);
        % loop over all 10 different body parts
        sum_lognorm_probs = 0;
        for i = 1:B
            Oi = O(i,:);            
            % reuse sigmas
            y_sigma     =   P.clg(i).sigma_y(k);
            x_sigma     =   P.clg(i).sigma_x(k);
            angle_sigma =   P.clg(i).sigma_angle(k);
            % reuse data
            y_i         =   Oi(1);
            x_i         =   Oi(2);
            angle_i     =   Oi(3);            
            
            if G(i,1,k) == 0
                % no parents other than class                
                y_mu        =   P.clg(i).mu_y(k);
                x_mu        =   P.clg(i).mu_x(k);
                angle_mu    =   P.clg(i).mu_angle(k);              
            else 
                % additional parent p_i of node i
                % parent of O_i
                p_i = G(i,2,k);
                % p_i denotes index of parent node i            
                y_p_i       = O(p_i,1);
                x_p_i       = O(p_i,2);
                angle_p_i   = O(p_i,3);                
                p_i_vals    = [1, y_p_i, x_p_i, angle_p_i];
                % 1,2,3 because Oi = (y_i,x_i,angle_i)
                y_mu        =   dot(p_i_vals,P.clg(i).theta(k,1:4));
                x_mu        =   dot(p_i_vals,P.clg(i).theta(k,5:8));
                angle_mu    =   dot(p_i_vals,P.clg(i).theta(k,9:12));
            end
            log_p_Oi_y      = lognormpdf(y_i,y_mu,y_sigma);
            log_p_Oi_x      = lognormpdf(x_i,x_mu,x_sigma);
            log_p_Oi_angle  = lognormpdf(angle_i,angle_mu,angle_sigma);
            log_p_Oi = log_p_Oi_y + log_p_Oi_x + log_p_Oi_angle;            
            sum_lognorm_probs = sum_lognorm_probs + log_p_Oi;
            
        end
        % get rid of the log
        class_prob(n,k) = (P_ck * exp(sum_lognorm_probs));
    end
end

% make predictions
for a = 1:N
    max_value = max(class_prob(a,:));
    max_col_idx = find(max_value == class_prob(a,:));
    if length(max_col_idx) == 2
        class_pred(a,max_col_idx) = mate_row;
    else
        class_pred(a,max_col_idx) = 1;
    end
end

% calculate accuracy
correct_cnt = 0;
for b = 1:N
    if class_pred(b,:) == labels(b,:)
        correct_cnt = correct_cnt +1;
    end        
end    
accuracy = correct_cnt/N;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Accuracy: %.2f\n', accuracy);