import numpy as np

def generate_synthetic_data_leakage(
    n,
    d,
    k,
    J,
    b,
    l,
    mu_x=None,
    sigma_x=None,
    sigma_c=None,
    sigma_c_hat=None,
    sigma_y=None,
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
    - y (ndarray): Target labels (n,). (zero-based indexing)
    """
    if seed is not None:
        np.random.seed(seed)  # Ensure reproducibility

    if k > b or k > d - b - l:
        print("Warning: k exceeds projection limits, resulting in reduced effective dimensionality.")
    if d < b:
        raise ValueError("Invalid configuration: b must be <= d.")
    
    # Set default mean/covariance matrices
    mu_x = mu_x if mu_x is not None else np.zeros(d)
    sigma_x = sigma_x if sigma_x is not None else np.eye(d)
    sigma_c = sigma_c if sigma_c is not None else np.eye(k)
    sigma_c_hat = sigma_c_hat if sigma_c_hat is not None else np.eye(k)
    sigma_y = sigma_y if sigma_y is not None else np.eye(J)

    # Generate feature matrix X
    X = np.random.multivariate_normal(mu_x, sigma_x, size=n)

    # Generate ground truth concepts
    A = np.hstack([np.random.normal(0, 1, size=(k, b)), np.zeros((k, d - b))])
    epsilon_c = np.random.multivariate_normal(np.zeros(k), sigma_c, size=n)
    logits_pi = np.dot(X, A.T) + epsilon_c
    pi_i = 1 / (1 + np.exp(-logits_pi))  # Sigmoid function
    c = np.random.binomial(1, pi_i)

    # Introduce leakage
    size_R_B = d - b - l
    B = np.zeros((k, d))
    B[:, b : b + size_R_B] = np.random.normal(0, 1, size=(k, size_R_B))
    l_i = np.dot(X, B.T)

    # Generate estimated concepts
    epsilon_c_hat = np.random.multivariate_normal(np.zeros(k), sigma_c_hat, size=n)
    logits_c_hat = np.dot(X, A.T) + l_i + epsilon_c_hat
    c_hat = 1 / (1 + np.exp(-logits_c_hat))  # Sigmoid function

    # Generate target logits
    if f is None:
        h = 10
        W1, W2 = np.random.normal(0, 1, (2 * k, h)), np.random.normal(0, 1, (h, J))
        b1, b2 = np.zeros(h), np.zeros(J)

        def f(c_i, l_i):
            hidden = np.maximum(0, np.dot(np.hstack([c_i, l_i]), W1) + b1)  # ReLU
            return np.dot(hidden, W2) + b2

    epsilon_y = np.random.multivariate_normal(np.zeros(J), sigma_y, size=n)
    logits_p = f(c, l_i) + epsilon_y
    p_i = np.exp(logits_p - logits_p.max(axis=1, keepdims=True))
    p_i /= p_i.sum(axis=1, keepdims=True)  # Softmax probabilities

    # Sample target labels (zero-based indexing)
    y = np.array([np.random.choice(J, p=p_i[i]) for i in range(n)])

    return X, c, c_hat, y