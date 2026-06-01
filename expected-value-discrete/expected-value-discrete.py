import numpy as np

def expected_value_discrete(x, p):
    """
    Compute the expected value of a discrete random variable.

    Parameters
    ----------
    x : array-like, shape (N,)
        Possible values of the random variable
    p : array-like, shape (N,)
        Corresponding probabilities (must sum to 1)

    Returns
    -------
    float
        Expected value E[X]
    """
    x = np.array(x, dtype=float)
    p = np.array(p, dtype=float)

    # Safety check: probabilities should sum to 1
    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("Probabilities must sum to 1.")

    return np.sum(x * p)
