function loglikelihood = ComputeLogLikelihood(P, G, dataset)
% returns the (natural) log-likelihood of data given the model and graph structure
%
% Inputs:
% P: struct array parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description)
%
%    NOTICE that G could be either 10x2 (same graph shared by all classes)
%    or 10x2x2 (each class has its own graph). your code should compute
%    the log-likelihood using the right graph.
%
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, angle)
% 
% Output:
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset,1); % number of examples
K = length(P.c); % number of classes
B = size(dataset,2); % number of body parts

loglikelihood = 0;
% You should compute the log likelihood of data as in eq. (12) and (13)
% in the PA description
% Hint: Use lognormpdf instead of log(normpdf) to prevent underflow.
%       You may use log(sum(exp(logProb))) to do addition in the original
%       space, sum(Prob).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
% determine if shared graph over all classes or individual graph for each
% class
if length(size(G)) == 2
    % replicate G for all K classes
    G = repmat(G,[1,1,K]);
end

% loop over N examples
for n = 1:N
    single_loglikelihood = 0;
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
        single_loglikelihood = single_loglikelihood + (P_ck * exp(sum_lognorm_probs));
    end
    loglikelihood = loglikelihood + log(single_loglikelihood);
    
end    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
