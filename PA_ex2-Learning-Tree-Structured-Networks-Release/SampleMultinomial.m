% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function sample = SampleMultinomial(probabilities)

dice = rand(1,1);
% rand returns a uniformly sampled random number from interval (0,1)

accumulate = 0;
for i=1:length(probabilities)
    accumulate = accumulate + probabilities(i);
   if accumulate/sum(probabilities) > dice
       break
   end
end
sample = i;


