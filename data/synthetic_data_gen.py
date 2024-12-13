import numpy as np

def generate_synthetic_data_leakage(
    n,
    d,
    k,
    J,
    b,
    l,
    mu_x=None,
    Sigma_x=None,
    Sigma_c=None,
    Sigma_c_hat=None,
    Sigma_y=None,
    f=None,
    seed=None,
):
    """
    Generate synthetic data with controlled leakage for concept bottleneck models.

    Parameters:
    - n (int): Number of observations.
    - d (int): Feature dimensionality.
    - k (int): Number of concepts.
    - J (int): Number of target classes.
    - b (int): Features used in ground truth concepts (k < b < d - k - l).
    - l (int): Features excluded from leakage (controls leakage).
    - Other params: Mean/covariance matrices, target function `f`, random seed.

    Returns:
    - X (ndarray): Feature matrix (n, d).
    - c (ndarray): Ground truth concepts (n, k).
    - c_hat (ndarray): Estimated concepts (n, k).
    - y (ndarray): Target labels (n,).
    """
    if seed is not None:
        np.random.seed(seed)  # Ensure reproducibility

    if k > b or k > d - b - l:
        print("Warning: k exceeds projection limits, resulting in reduced effective dimensionality.")
    if d < b:
        raise ValueError("Invalid configuration: b must be <= d.")
    
    # Set default mean/covariance matrices
    mu_x = mu_x if mu_x is not None else np.zeros(d)
    Sigma_x = Sigma_x if Sigma_x is not None else np.eye(d)
    Sigma_c = Sigma_c if Sigma_c is not None else np.eye(k)
    Sigma_c_hat = Sigma_c_hat if Sigma_c_hat is not None else np.eye(k)
    Sigma_y = Sigma_y if Sigma_y is not None else np.eye(J)

    # Generate feature matrix X
    X = np.random.multivariate_normal(mu_x, Sigma_x, size=n)

    # Generate ground truth concepts
    A = np.hstack([np.random.normal(0, 1, size=(k, b)), np.zeros((k, d - b))])
    epsilon_c = np.random.multivariate_normal(np.zeros(k), Sigma_c, size=n)
    logits_pi = np.dot(X, A.T) + epsilon_c
    pi_i = 1 / (1 + np.exp(-logits_pi))  # Sigmoid function
    c = np.random.binomial(1, pi_i)

    # Introduce leakage
    size_R_B = d - b - l
    B = np.zeros((k, d))
    B[:, b : b + size_R_B] = np.random.normal(0, 1, size=(k, size_R_B))
    l_i = np.dot(X, B.T)

    # Generate estimated concepts
    epsilon_c_hat = np.random.multivariate_normal(np.zeros(k), Sigma_c_hat, size=n)
    logits_c_hat = np.dot(X, A.T) + l_i + epsilon_c_hat
    c_hat = 1 / (1 + np.exp(-logits_c_hat))  # Sigmoid function

    # Generate target logits
    if f is None:
        h, W1, W2 = 10, np.random.normal(0, 1, (2 * k, h)), np.random.normal(0, 1, (h, J))
        b1, b2 = np.zeros(h), np.zeros(J)

        def f(c_i, l_i):
            hidden = np.maximum(0, np.dot(np.hstack([c_i, l_i]), W1) + b1)  # ReLU
            return np.dot(hidden, W2) + b2

    epsilon_y = np.random.multivariate_normal(np.zeros(J), Sigma_y, size=n)
    logits_p = f(c, l_i) + epsilon_y
    p_i = np.exp(logits_p - logits_p.max(axis=1, keepdims=True))
    p_i /= p_i.sum(axis=1, keepdims=True)  # Softmax probabilities

    # Sample target labels
    y = np.array([np.random.choice(J, p=p_i[i]) for i in range(n)]) + 1

    return X, c, c_hat, y

# Example usage
# n, d, k, J, b, l = 1000, 50, 10, 5, 20, 10
# X, c, c_hat, y = generate_synthetic_data_leakage(n, d, k, J, b, l, seed=42)
