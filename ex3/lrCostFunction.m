function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

regTerm = lambda/(2*m) * sum(theta(2:end).^2);

hypothesis = sigmoid(X * theta);
J = sum((-y .* log(hypothesis)-(1-y).*log(1-hypothesis)) + regTerm)/m;
grad = ( (X' * (hypothesis - y)) + (lambda * [0; theta(2:end)]) )/m ;

% not fully vectorized code from the last linear regression assignment
% 2013/05/13 - i should change the implementation for those
% earlier assingments - johannes

% for i = 1:m
%     hypothesis = sigmoid(theta' * X(i,:)');
%     J = J +(-y(i) * log(hypothesis)) - (1-y(i))*log(1 - hypothesis);
%     grad = grad + (hypothesis - y(i)) * X(i,:)';
% end
% J = J/m;
% grad = grad/m;
% 
% regTerm = lambda/(2*m) * dot(theta(2:size(theta)), theta(2:size(theta)));
% 
% for i = 1:m
%     hypothesis = sigmoid(theta' * X(i,:)');
%     J = J +(-y(i) * log(hypothesis)) - (1-y(i))*log(1 - hypothesis) + regTerm;
%     grad = grad + (hypothesis - y(i)) * X(i,:)';
% end
% 
% J = J/m;
% 
% grad = grad/m + lambda * [0; theta(2:size(theta))] / m;

% =============================================================

grad = grad(:);

end
