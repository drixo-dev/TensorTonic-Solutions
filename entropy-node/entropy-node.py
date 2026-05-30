import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    # Count occurrences of each class
    values, counts = np.unique(y, return_counts=True)
    
    # Compute probabilities
    probabilities = counts / counts.sum()
    
    # Use np.log2 with masking to avoid log(0)
    entropy = -np.sum(probabilities * np.log2(probabilities, where=(probabilities > 0)))
    
    return entropy
    pass