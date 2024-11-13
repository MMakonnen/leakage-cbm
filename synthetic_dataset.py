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
    sigma_hat_c=None,
    sigma_y=None,
    f=None,
    seed=None,
):
    """
    Generate synthetic data with controlled leakage for concept bottleneck models.

    Parameters:
    - n (int): Number of observations.
    - d (int): Dimensionality of features.
    - k (int): Number of concepts.
    - J (int): Number of target classes.
    - b (int): Number of features used in ground truth concepts (k < b < d - k - l).
    - l (int): Number of features excluded from leakage (controls leakage amount).
    - mu_x (array-like): Mean vector of the feature distribution (default zeros).
    - sigma_x (array-like): Covariance matrix of the feature distribution (default identity).
    - sigma_c (array-like): Covariance matrix of noise in ground truth concepts (default identity).
    - sigma_hat_c (array-like): Covariance matrix of noise in estimated concepts (default identity).
    - sigma_y (array-like): Covariance matrix of noise in target logits (default identity).
    - f (callable): Function to compute target logits from concepts and leakage (default MLP).
    - seed (int): Random seed for replicability.

    Returns:
    - X (ndarray): Features matrix of shape (n, d).
    - c (ndarray): Ground truth concepts matrix of shape (n, k).
    - hat_c (ndarray): Estimated concepts matrix of shape (n, k).
    - y (ndarray): Target labels array of shape (n,).
    """
    # Set random seed for replicability
    if seed is not None:
        np.random.seed(seed)

    # Set default values for mean and covariance matrices if not provided
    if mu_x is None:
        mu_x = np.zeros(d)
    if sigma_x is None:
        sigma_x = np.eye(d)
    if sigma_c is None:
        sigma_c = np.eye(k)
    if sigma_hat_c is None:
        sigma_hat_c = np.eye(k)
    if sigma_y is None:
        sigma_y = np.eye(J)

    # Generate features X ~ N(mu_x, sigma_x)
    X = np.random.multivariate_normal(mu_x, sigma_x, size=n)  # Shape: (n, d)

    # Construct matrix A for ground truth concepts
    R_A = np.random.normal(0, 1, size=(k, b))  # Shape: (k, b)
    A = np.hstack([R_A, np.zeros((k, d - b))])  # Shape: (k, d)

    # Generate noise for ground truth concepts
    epsilon_c = np.random.multivariate_normal(np.zeros(k), sigma_c, size=n)  # Shape: (n, k)

    # Compute logits for pi_i
    logits_pi = np.dot(X, A.T) + epsilon_c  # Shape: (n, k)

    # Compute success probabilities pi_i using the sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    pi_i = sigmoid(logits_pi)  # Shape: (n, k)

    # Sample ground truth concepts c_i ~ Bernoulli(pi_i)
    c = np.random.binomial(1, pi_i)  # Shape: (n, k)

    # Construct matrix B for leakage
    size_R_B = d - b - l
    R_B = np.random.normal(0, 1, size=(k, size_R_B))  # Shape: (k, d - b - l)
    B = np.zeros((k, d))  # Initialize B with zeros
    B[:, b : d - l] = R_B  # Fill in the appropriate columns

    # Compute leakage term l_i = B x_i
    l = np.dot(X, B.T)  # Shape: (n, k)

    # Generate noise for estimated concepts
    epsilon_hat_c = np.random.multivariate_normal(np.zeros(k), sigma_hat_c, size=n)  # Shape: (n, k)

    # Compute logits for estimated concepts hat_c_i
    logits_hat_c = np.dot(X, A.T) + l + epsilon_hat_c  # Shape: (n, k)

    # Compute estimated concepts hat_c_i using the sigmoid function
    hat_c = sigmoid(logits_hat_c)  # Shape: (n, k)

    # Generate noise for target logits
    epsilon_y = np.random.multivariate_normal(np.zeros(J), sigma_y, size=n)  # Shape: (n, J)

    # Define default function f if not provided
    if f is None:
        # Simple MLP with one hidden layer
        h = 10  # Hidden layer size
        W1 = np.random.normal(0, 1, size=(2 * k, h))
        b1 = np.zeros(h)
        W2 = np.random.normal(0, 1, size=(h, J))
        b2 = np.zeros(J)

        def f(c_i, l_i):
            inputs = np.hstack([c_i, l_i])  # Shape: (n, 2k)
            hidden = np.maximum(0, np.dot(inputs, W1) + b1)  # ReLU activation
            outputs = np.dot(hidden, W2) + b2  # Shape: (n, J)
            return outputs

    # Compute logits for targets
    logits_p = f(c, l) + epsilon_y  # Shape: (n, J)

    # Compute target probabilities using the softmax function
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
        return e_x / e_x.sum(axis=1, keepdims=True)

    p_i = softmax(logits_p)  # Shape: (n, J)

    # Sample target labels y_i ~ Categorical(p_i)
    y = np.array([np.random.choice(J, p=p_i[i]) for i in range(n)]) + 1  # Labels from 1 to J

    return X, c, hat_c, y


# EXAMPLE USAGE OF ABOVE FUNCTION

n = 1000   # Number of observations
d = 50     # Number of features
k = 10     # Number of concepts
J = 5      # Number of target classes
b = 15     # Number of features used in ground truth concepts
l = 10     # Number of features excluded from leakage

X, c, hat_c, y = generate_synthetic_data_leakage(n, d, k, J, b, l, seed=42)
