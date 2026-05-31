import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)              # Shape (N, D)
    y = np.array(y)              # Shape (N,)
    N, D = X.shape
    
    # Initialize parameters
    w = np.zeros(D)
    b = 0.0
    
    for _ in range(steps):
        # Linear combination
        z = X @ w + b
        # Predictions
        p = _sigmoid(z)
        
        # Gradients
        grad_w = (1/N) * (X.T @ (p - y))
        grad_b = (1/N) * np.sum(p - y)
        
        # Update parameters
        w -= lr * grad_w
        b -= lr * grad_b
    
    return w, b
