function [theta, J_history] = untitled1(X, y, theta, alpha, n)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, n) updates theta by 
%   taking n gradient steps with learning rate alpha

% Initialize some useful values
X = [1 1 1;1 2 3;1 3 4];
y = [1;2;3];
theta = [0 1 2];
alpha = 0.01;
n = 15;
m = length(y); % number of training examples
J_history = zeros(n, 1);

for j = 1:n

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    prediction = theta' * X(j,:);
    
    delta = (1/m)*(sum(prediction(j) - y(j)) .* X(j,:));

    theta(j) = theta(j) - alpha * delta(j);





    % ============================================================

    % Save the cost J in every jation    
    J_history(j) = computeCost(X, y, theta);

end

end