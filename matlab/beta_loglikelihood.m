function out = beta_loglikelihood(theta,x)

%     Function equal to -log L(\theta;x) to be fed to fminsearch
% 
%     Parameters
%     ----------
%     theta: theta(1) is alpha and theta(2) is beta
%     x: x is the data

    a = theta(1);
    b = theta(2);
    n = length(x);
    
    % Log-likelihood
    obj = (a - 1) * sum(log(x)) + (b - 1) * sum(log(1 - x)) - n * log(beta(a, b)) ;
    % We want to maximize
    sense = -1 ;

    out = sense * obj;
end