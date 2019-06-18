% function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)
% returns the negative log-likelihood and its gradient, given a CRF with parameters theta,
% on data (X, y). 
%
% Inputs:
% X            Data.                           (numCharacters x numImageFeatures matrix)
%              X(:,1) is all ones, i.e., it encodes the intercept/bias term.
% y            Data labels.                    (numCharacters x 1 vector)
% theta        CRF weights/parameters.         (numParams x 1 vector)
%              These are shared among the various singleton / pairwise features.
% modelParams  Struct with three fields:
%   .numHiddenStates     in our case, set to 26 (26 possible characters)
%   .numObservedStates   in our case, set to 2  (each pixel is either on or off)
%   .lambda              the regularization parameter lambda
%
% Outputs:
% nll          Negative log-likelihood of the data.    (scalar)
% grad         Gradient of nll with respect to theta   (numParams x 1 vector)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)

    % featureSet is a struct with two fields:
    %    .numParams - the number of parameters in the CRF (this is not numImageFeatures
    %                 nor numFeatures, because of parameter sharing)
    %    .features  - an array comprising the features in the CRF.
    %
    % Each feature is a binary indicator variable, represented by a struct 
    % with three fields:
    %    .var          - a vector containing the variables in the scope of this feature
    %    .assignment   - the assignment that this indicator variable corresponds to
    %    .paramIdx     - the index in theta that this feature corresponds to
    %
    % For example, if we have:
    %   
    %   feature = struct('var', [2 3], 'assignment', [5 6], 'paramIdx', 8);
    %
    % then feature is an indicator function over X_2 and X_3, which takes on a value of 1
    % if X_2 = 5 and X_3 = 6 (which would be 'e' and 'f'), and 0 otherwise. 
    % Its contribution to the log-likelihood would be theta(8) if it's 1, and 0 otherwise.
    %
    % If you're interested in the implementation details of CRFs, 
    % feel free to read through GenerateAllFeatures.m and the functions it calls!
    % For the purposes of this assignment, though, you don't
    % have to understand how this code works. (It's complicated.)
    
    featureSet = GenerateAllFeatures(X, modelParams);

    % Use the featureSet to calculate nll and grad.
    % This is the main part of the assignment, and it is very tricky - be careful!
    % You might want to code up your own numerical gradient checker to make sure
    % your answers are correct.
    %
    % Hint: you can use CliqueTreeCalibrate to calculate logZ effectively. 
    %       We have halfway-modified CliqueTreeCalibrate; complete our implementation 
    %       if you want to use it to compute logZ.        
    nll = 0;
    grad = zeros(size(theta));
    %%%
    % Your code here:
    
factornum=1;
factors(1)=struct('var',featureSet.features(1).var,'card',modelParams.numHiddenStates*ones(1,length(featureSet.features(1).var)),'val',zeros(1,modelParams.numHiddenStates^length(featureSet.features(1).var)));
L=length(featureSet.features);
for i=1:L
    factorbelong=0;
    for j=1:factornum
        if (length(featureSet.features(i).var)==length(factors(j).var)) && (sum(featureSet.features(i).var==factors(j).var)==length(factors(j).var))
            factorbelong=j;
        end
    end
    if factorbelong==0
        factornum=factornum+1;
        factors(factornum)=struct('var',featureSet.features(i).var,'card',modelParams.numHiddenStates*ones(1,length(featureSet.features(i).var)),'val',zeros(1,modelParams.numHiddenStates^length(featureSet.features(i).var)));
        factorbelong=factornum;
    end
    indx=AssignmentToIndex(featureSet.features(i).assignment,modelParams.numHiddenStates*ones(1,length(featureSet.features(i).var)));
    factors(factorbelong).val(indx)=factors(factorbelong).val(indx)+theta(featureSet.features(i).paramIdx);
end
for i=1:factornum
    for j=1:length(factors(i).val)
        factors(i).val(j)=exp(factors(i).val(j));
    end
end
cliquetree=CreateCliqueTree(factors);
[calibrated,logZ]=CliqueTreeCalibrate(cliquetree,0);

%weighted feature counts
weightedfc=zeros(1,featureSet.numParams);
datafc=zeros(1,featureSet.numParams);
for i=1:L
    countthat=1;
    for j=1:length(featureSet.features(i).var)
        if y(featureSet.features(i).var(j))~=featureSet.features(i).assignment(j)
            countthat=0;
        end
    end
    if countthat==1
        weightedfc(featureSet.features(i).paramIdx)=weightedfc(featureSet.features(i).paramIdx)+theta(featureSet.features(i).paramIdx);
        datafc(featureSet.features(i).paramIdx)=datafc(featureSet.features(i).paramIdx)+1;
    end
end

%model feature counts
modelfc=zeros(1,featureSet.numParams);
for i=1:length(calibrated.cliqueList)
    calibrated.cliqueList(i).val=calibrated.cliqueList(i).val/sum(calibrated.cliqueList(i).val);
end
for i=1:L
    for j=1:length(calibrated.cliqueList)
        countx=0;
        for m=1:length(featureSet.features(i).var)
            for n=1:length(calibrated.cliqueList(j).var)
                if featureSet.features(i).var(m)==calibrated.cliqueList(j).var(n)
                    countx=countx+1;
                end
            end
        end
        if countx==length(featureSet.features(i).var)
            usethat=j;
            break;
        end
    end
    if length(featureSet.features(i).var)==length(calibrated.cliqueList(usethat).var)
        marg=calibrated.cliqueList(usethat);
    else
        sumout=setdiff(calibrated.cliqueList(usethat).var,featureSet.features(i).var);
        marg=FactorMarginalization(calibrated.cliqueList(usethat),sumout);
    end
    %marg.val=marg.val/sum(marg.val);
    indx=AssignmentToIndex(featureSet.features(i).assignment,marg.card);
    modelfc(featureSet.features(i).paramIdx)=modelfc(featureSet.features(i).paramIdx)+marg.val(indx);
end

%last step
nll=logZ-sum(weightedfc)+(modelParams.lambda)/2*sum(theta.*theta);
grad=modelfc-datafc+(modelParams.lambda)*theta;
    

end
