% File: RecognizeActions.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [accuracy, predicted_labels] = RecognizeActions(datasetTrain, datasetTest, G, maxIter)

% INPUTS
% datasetTrain: dataset for training models, see PA for details
% datasetTest: dataset for testing models, see PA for details
% G: graph parameterization as explained in PA decription
% maxIter: max number of iterations to run for EM

% OUTPUTS
% accuracy: recognition accuracy, defined as (#correctly classified examples / #total examples)
% predicted_labels: N x 1 vector with the predicted labels for each of the instances in datasetTest, with N being the number of unknown test instances


% Train a model for each action
% Note that all actions share the same graph parameterization and number of max iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nModels = length(datasetTrain);
HMM = repmat(struct("P",[],"loglikelihood",[],"ClassProb",[],"PairProb",[]),4,1);
for i = 1:nModels
    actionData_i = datasetTrain(i).actionData;
    poseData_i = datasetTrain(i).poseData;
    InitialClassProb_i = datasetTrain(i).InitialClassProb;
    InitialPairProb_i = datasetTrain(i).InitialPairProb;
    [HMM(i).P, HMM(i).loglikelihood, HMM(i).ClassProb, HMM(i).PairProb] = ...
    EM_HMM(actionData_i, poseData_i, G, InitialClassProb_i, InitialPairProb_i, maxIter);
end    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Classify each of the instances in datasetTrain
% Compute and return the predicted labels and accuracy
% Accuracy is defined as (#correctly classified examples / #total examples)
% Note that all actions share the same graph parameterization

accuracy = 0;
predicted_labels = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nTestActions = length(datasetTest.actionData);
predicted_labels = zeros(nTestActions,1);
% loop over each action
for j = 1:nTestActions
    model_LLs = zeros(nModels,1);
    % loop over each action specific candidate model
    for m = 1:nModels
        actionData = datasetTest.actionData(j);
        poseData_idx = datasetTest.actionData(j).marg_ind;
        poseData = datasetTest.poseData(poseData_idx,:,:);
        
        N = size(poseData,1); % number of examples for current action j
        B = size(poseData,2); % number of body parts
        K = size(HMM(m).ClassProb,2); % number of classes
        P = HMM(m).P;
        
        logEmissionProb = zeros(N, K);
        
        % loop over N examples
        for n = 1:N
            % fetch current sample O
            O = poseData(n,:,:);
            % O has size: 1x10x3 thus remove redundant dimensions of length 1
            O = squeeze(O);
            % loop over K classes
            for k = 1:K
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
                    
                    if G(i,1) == 0
                        % no parents other than class
                        y_mu        =   P.clg(i).mu_y(k);
                        x_mu        =   P.clg(i).mu_x(k);
                        angle_mu    =   P.clg(i).mu_angle(k);
                    else
                        % additional parent p_i of node i
                        % parent of O_i
                        p_i = G(i,2);
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
                logEmissionProb(n,k) = sum_lognorm_probs;
            end
        end

        % fill in factors for each state S_i by computing all 3 CPDs
        % P(S_1) , P(S_i|S_i-1), P(P_j|S_j) -> cf. page 6 of PA .pdf
        % in total we have N poses/states -> 2*N factor entries
        factorList = repmat(struct ('var',[],'card',[],'val',[]),1,2*N);
        factor_idx = 1;
        
        % init state prior P(S_1)
        factorList(factor_idx).var = 1;
        factorList(factor_idx).card = K;
        factorList(factor_idx).val = log(P.c);
        % transition factors P(S_i|S_i-1)
        factor_idx = 2;
        for i = 2:N
            factorList(factor_idx).var = [i, i-1];
            factorList(factor_idx).card = [K, K];
            factorList(factor_idx).val = log(P.transMatrix(:)');
            % increase factor_idx
            factor_idx = factor_idx + 1;
        end
        % emission model probs P(P_j|S_j) reduced by observed pose P_j
        % -> theta'(S_j)
        for i = 1:N
            factorList(factor_idx).var = i;
            factorList(factor_idx).card = K;
            factorList(factor_idx).val = logEmissionProb(i,:);
            factor_idx = factor_idx + 1;
        end
        % run clique tree inference calibration in log-space
        [~, PCalibrated] = ComputeExactMarginalsHMM(factorList);
        model_LLs(m) = logsumexp(PCalibrated.cliqueList(end).val);
    end
    [~, predicted_labels(j)] = max(model_LLs); 
end
% evaluation: binary equality vector
binary_check = predicted_labels == datasetTest.labels;
accuracy = sum(binary_check)/nTestActions;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
