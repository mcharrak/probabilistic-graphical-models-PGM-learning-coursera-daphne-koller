% File: EM_cluster.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P, loglikelihood, ClassProb] = EM_cluster(poseData, G, InitialClassProb, maxIter)

% INPUTS
% poseData: N x 10 x 3 matrix, where N is number of poses;
%   poseData(i,:,:) yields the 10x3 matrix for pose i.
% G: graph parameterization as explained in PA8
% InitialClassProb: N x K, initial allocation of the N poses to the K
%   classes. InitialClassProb(i,j) is the probability that example i belongs
%   to class j
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K, conditional class probability of the N examples to the
%   K classes in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to class j

% Initialize variables
N = size(poseData, 1); % number of poses in dataset
K = size(InitialClassProb, 2); % # of classes (e.g. actions "clap", "high_kick")
B = size(poseData,2); % B... # of body parts

ClassProb = InitialClassProb;

loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg.sigma_x = [];
P.clg.sigma_y = [];
P.clg.sigma_angle = [];

% EM algorithm
for iter=1:maxIter
  
  % M-STEP to estimate parameters for Gaussians
  %
  % Fill in P.c with the estimates for prior class probabilities
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  %
  % Hint: This part should be similar to your work from PA8
  
  P.c = zeros(1,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
  % 1. determine if shared graph over all classes or individual graph for each
  % class
  if length(size(G)) == 2
    % replicate G for all K classes
    G = repmat(G,[1,1,K]);
  end

  % 2. calculate P.c (as normalized counts for each class) using MLE
  for k = 1:K
      sum_k = sum(ClassProb(:,k));
      P.c(k) = sum_k/N;
  end
  
  % 3. calculate P.clg
  % loop over each bodypart
  for i = 1:B
      % pick relevant data for current body part i
      data_y = poseData(:,i,1);              
      data_x = poseData(:,i,2);      
      data_angle = poseData(:,i,3);      
      % loop over each class k
      for k = 1:K
          % now every example will be considered for every class k because 
          % we have weights for each class for each example
          % case that node i has no parent other than class label
          if G(i,1,k) == 0             
              [mu_y, sigma_y] = FitG(data_y,ClassProb(:,k));
              [mu_x, sigma_x] = FitG(data_x,ClassProb(:,k));
              [mu_angle, sigma_angle] = FitG(data_angle,ClassProb(:,k));
              % populate mus
              P.clg(i).mu_y(1,k) = mu_y;
              P.clg(i).mu_x(1,k) = mu_x;
              P.clg(i).mu_angle(1,k) = mu_angle;
              % populate sigmas
              P.clg(i).sigma_y(1,k) = sigma_y;
              P.clg(i).sigma_x(1,k) = sigma_x;
              P.clg(i).sigma_angle(1,k) = sigma_angle;
              % populate theta
              % leave theta empty
          % case that node i has additional parent
          else
              % determine idx of parent for node i
              p_i = G(i,2,k);
              data_parent = poseData(:,p_i,:);
              % data_parent has size Nx1x3; need to remove dim of len 1
              data_parent = squeeze(data_parent);
              % y
              [Beta_y, sigma_y] = FitLG(data_y,data_parent,ClassProb(:,k));
              % x
              [Beta_x, sigma_x] = FitLG(data_x,data_parent,ClassProb(:,k));
              % angle
              [Beta_angle, sigma_angle] = FitLG(data_angle,data_parent,ClassProb(:,k));
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
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % E-STEP to re-estimate ClassProb using the new parameters
  %
  % Update ClassProb with the new conditional class probabilities.
  % Recall that ClassProb(i,j) is the probability that example i belongs to
  % class j.
  %
  % You should compute everything in log space, and only convert to
  % probability space at the end.
  %
  % Tip: To make things faster, try to reduce the number of calls to
  % lognormpdf, and inline the function (i.e., copy the lognormpdf code
  % into this file)
  %
  % Hint: You should use the logsumexp() function here to do
  % probability normalization in log space to avoid numerical issues
  
  ClassProb = zeros(N,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for n = 1:N
      % fetch current sample O
      O = poseData(n,:,:);
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
              % Computes natural-logarithm of the normal pdf inline instead
              % of making the function call: lognormpdf(arg1,arg2,arg3)
              log_p_Oi_y = -log(y_sigma*sqrt(2*pi))-(y_i-y_mu).^2 ./ ...
                  (2*y_sigma.^2);
              log_p_Oi_x = -log(x_sigma*sqrt(2*pi))-(x_i-x_mu).^2 ./ ...
                  (2*x_sigma.^2); 
              log_p_Oi_angle = -log(angle_sigma*sqrt(2*pi))-(angle_i-angle_mu).^2 ./ ...
                  (2*angle_sigma.^2);               

              log_p_Oi = log_p_Oi_y + log_p_Oi_x + log_p_Oi_angle;
              sum_lognorm_probs = sum_lognorm_probs + log_p_Oi;
              
          end
          % stay in log-space
          ClassProb(n,k) = log(P_ck) + sum_lognorm_probs;
      end
  end
  
  % now we need to normalize the probabilities in ClassProb which are in
  % log space. In regular prob-space we can simply sum probs in of each row
  % BUT as we are in log space, we need to 1. apply exp() on each element
  % in a row, 2. sum up the exp(log_prob) terms 3. convert back to
  % log-space by applying log() --> this is implemented within the given helper
  % function logsumexp()
  
  % calculate normalizing vector in log-space for each of the N poses
  norm_vec = logsumexp(ClassProb);
  % now normalize each value log_prob value in ClassProb
  % in prob-space: we divide each prob by the sum of all probs in its row
  % in log-space: we substract each log_prob by the logsumexp term of its row
  ClassProb = ClassProb - norm_vec;
  
  %finally convert ClassProb into prob-space
  ClassProb = exp(ClassProb);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Compute log likelihood of dataset for this iteration
  % Hint: You should use the logsumexp() function here
  loglikelihood(iter) = 0;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % the log-likelihood can be computed in two steps:
  % 1. sum of all probs in ClassProb
  % 2. log of the sum from 1.
  % --> unfortunately, this will lead to overflows because the 
  % but because the probs are so small, we do experience underflow
  % INSTEAD simply add up the values from the earlier calculated row sums
  loglikelihood(iter) = sum(norm_vec);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Print out loglikelihood
  fprintf('EM iteration %d: log likelihood: %f\n', ...
    iter, loglikelihood(iter));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
  
  % Check for overfitting: when loglikelihood decreases
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);


%%%%%%%%%%%%%%%%%%%%%%%% BELOW WRONG APPROACHES %%%%%%%%%%%%%%%%%%%%%%%%

% % 2. calculate P.c (as normalized counts for each class) using MLE
% [~,max_cols] = max(InitialClassProb,[],2);
% % labels contains the hard assignments
% labels = full(ind2vec(max_cols')');
% for k = 1:K
%     count_k = sum(labels(:,k));
%     P.c(k) = count_k/N;
% end
