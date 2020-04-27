function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

n = length(X(1,:));
prediction = 1 ./ (1+e.^(-1 * (X * theta)));
t1 = 0;
t2 = 0;

for i = 1:m
    t1 += -1*(1/m)*((y(i)*log(prediction(i))) + (1-y(i))*(log(1 - prediction(i)))); 
end;



for i = 2:n 
	t2 += (lambda/(2*m))*(theta(i)^2);
end;


     
J = t1 + t2;

reg = ((lambda/m) * theta(2:n));

theta(1) = 0;

thetach = [theta(1);reg];


grad = ((1/m)*((prediction - y)' * X))' + thetach;






% =============================================================

end
