% File: EM_HMM.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P, loglikelihood, ClassProb, PairProb] = EM_HMM(actionData, poseData, G, InitialClassProb, InitialPairProb, maxIter)

% INPUTS
% actionData: structure holding the actions as described in the PA
% poseData: N x 10 x 3 matrix, where N is number of poses in all actions
% G: graph parameterization as explained in PA description
% InitialClassProb: N x K matrix, initial allocation of the N poses to the K
%   states. InitialClassProb(i,j) is the probability that example i belongs
%   to state j.
%   This is described in more detail in the PA.
% InitialPairProb: V x K^2 matrix, where V is the total number of pose
%   transitions in all HMM action models, and K is the number of states.
%   This is described in more detail in the PA.
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K matrix of the conditional class probability of the N examples to the
%   K states in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to state j. This is described in more detail in the PA.
% PairProb: V x K^2 matrix, where V is the total number of pose transitions
%   in all HMM action models, and K is the number of states. This is
%   described in more detail in the PA.

% Initialize variables
N = size(poseData, 1); % number of examples
B = size(poseData,2); % number of body parts
K = size(InitialClassProb, 2); % number of action classes (here 3)
L = size(actionData, 2); % number of actions
V = size(InitialPairProb, 1); % number of edges in chain of all states S_i

ClassProb = InitialClassProb; % cond. probs that pose i belongs to STATE! k, NOT class K
PairProb = InitialPairProb; % pairwise trans. prob. for pairs of consecutive states S_i

loglikelihood = zeros(maxIter,1);

P.c = []; % initial state prior prob. i.e. prob. of initial state belonging to class 1, 2 or 3
P.clg.sigma_x = [];
P.clg.sigma_y = [];
P.clg.sigma_angle = [];

% EM algorithm
for iter=1:maxIter
  
  % M-STEP to estimate parameters for Gaussians
  % Fill in P.c, the initial state prior probability (NOT the class probability as in PA8 and EM_cluster.m)
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  % Hint: This part should be similar to your work from PA8 and EM_cluster.m
  
  P.c = zeros(1,K); % init state/pose class prior probs instead of class prior prob for all N poses
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
  % 1. determine if shared graph over all classes or individual graph for each
  % class
  if length(size(G)) == 2
      % replicate G for all K classes
      G = repmat(G,[1,1,K]);
  end
  
  % 2. calculate P.c as follow:
  % add up the class probs. for the initial pose in each of the L actions
  sum_init_state_ClassProb = zeros(1,K);
  for l = 1:L
      curr_action_init_pose_idx = actionData(l).marg_ind(1);
      curr_init_state_ClassProb = ClassProb(curr_action_init_pose_idx,:);
      sum_init_state_ClassProb = sum_init_state_ClassProb + curr_init_state_ClassProb; 
  end
  % now normalize the prob by the number of actions
  P.c = sum_init_state_ClassProb/L;
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
  
  % M-STEP to estimate parameters for transition matrix
  % Fill in P.transMatrix, the transition matrix for states
  % P.transMatrix(i,j) is the probability of transitioning from state i to state j
  P.transMatrix = zeros(K,K);
  
  % Add Dirichlet prior based on size of poseData to avoid 0 probabilities
  P.transMatrix = P.transMatrix + size(PairProb,1) * .05;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % 4. pairProb contains the state transition probs.
  sum_S_ij_trans_prob = sum(PairProb,1);
  % reshape to quadratic transition matrix
  unnorm_transMatrix = reshape(sum_S_ij_trans_prob,K,K);
  % Add Dirichlet prior based on size of poseData to avoid 0 probabilities
  transMatrix = unnorm_transMatrix + size(PairProb,1) * .05;
  % normalize because in MC transition matrix the sum of outgoing probs = 1
  % rows represent outgoing probs
  P.transMatrix = transMatrix ./ sum(transMatrix,2);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    
  % E-STEP preparation: compute the emission model factors (emission probabilities) in log space for each 
  % of the poses in all actions = log( P(Pose | State) )
  % Hint: This part should be similar to (but NOT the same as) your code in EM_cluster.m
  
  logEmissionProb = zeros(N,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % 5. compute emission probabilities in log-space (i.e. P(pose|state) )
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
          logEmissionProb(n,k) = sum_lognorm_probs;
      end
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    
  % E-STEP to compute expected sufficient statistics
  % ClassProb contains the conditional class probabilities for each pose in all actions
  % PairProb contains the expected sufficient statistics for the transition CPDs (pairwise transition probabilities)
  % Also compute log likelihood of dataset for this iteration
  % You should do inference and compute everything in log space, only converting to probability space at the end
  % Hint: You should use the logsumexp() function here to do probability normalization in log space to avoid numerical issues
  
  ClassProb = zeros(N,K);
  PairProb = zeros(V,K^2);
  loglikelihood(iter) = 0;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % 6. compute factorlist F w/ singleton and pairwise factors for each
  % of the L actions
  for l = 1:L
      numPoses_action_i = length(actionData(l).marg_ind);
      numEdges_action_i = length(actionData(l).pair_ind);
      % create empty factorList
      factorList = repmat(struct('var',[],'card',[],'val',[]), ...
                    1,1);
      % fill in initial state
      factorList(1).var = 1;
      factorList(1).card = K;
      % we need log-space b/c we want to use logEmissionProb from step 5.
      factorList(1).val = log(P.c);
      for f = 1:numPoses_action_i
          factorList(f+1).var = f;
          factorList(f+1).card = K;
          factorList(f+1).val = logEmissionProb(actionData(l).marg_ind(f),:);          
      end
      len_FactorList = length(factorList);
      for f = 1:numEdges_action_i
          factorList(len_FactorList+f).var = [f,f+1]; % consecutive states S_i and S_j
          factorList(len_FactorList+f).card = [K,K];
          % move from prob-space to log-space
          factorList(len_FactorList+f).val = log(P.transMatrix(:)');
      end
      % run clique tree inference in log-space with helper function where 
      % M(i) represents the ith variable and M(i).val represents the
      % marginals of the ith variable
      [M, PCalibrated] = ComputeExactMarginalsHMM(factorList);
      for f=1:numPoses_action_i
        ClassProb(actionData(l).marg_ind(M(f).var),1:K) = M(f).val;
      end
      for f=1:numEdges_action_i
        PairProb(actionData(l).pair_ind(f),:) = PCalibrated.cliqueList(f).val;
      end
      loglikelihood(iter) = loglikelihood(iter) + logsumexp(PCalibrated.cliqueList(end).val);
  end
  % calculate normalizing constants
  norm_vec_ClassProb = logsumexp(ClassProb);
  norm_vec_PairProb = logsumexp(PairProb);
  % normalize in log-space
  ClassProb = ClassProb - norm_vec_ClassProb;
  PairProb = PairProb - norm_vec_PairProb;
  % convert back to prob-space
  ClassProb=exp(ClassProb);
  PairProb=exp(PairProb);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Print out loglikelihood
  disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
  
  % Check for overfitting by decreasing loglikelihood
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);
