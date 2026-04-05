def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    # Input validation
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.shape[0] != X.shape[0]:
        raise ValueError("Number of samples in X and y must match")
    if not isinstance(lam, (int, float)) or lam < 0:
        raise ValueError("lambda must be a non-negative number")

    n_features = X.shape[1]

    # Identity matrix (no regularization on bias term if bias is included separately)
    I = np.eye(n_features)

    # Ridge Regression closed-form solution
    beta = np.linalg.inv(X.T @ X + lam * I) @ X.T @ y

    return beta.flatten()
