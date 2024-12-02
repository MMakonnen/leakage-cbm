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
    - d (int): Dimensionality of features.
    - k (int): Number of concepts.
    - J (int): Number of target classes.
    - b (int): Number of features used in ground truth concepts (k < b < d - k - l).
    - l (int): Number of features excluded from leakage (controls leakage amount).
    - mu_x (array-like): Mean vector of the feature distribution (default zeros).
    - Sigma_x (array-like): Covariance matrix of the feature distribution (default identity).
    - Sigma_c (array-like): Covariance matrix of noise in ground truth concepts (default identity).
    - Sigma_c_hat (array-like): Covariance matrix of noise in estimated concepts (default identity).
    - Sigma_y (array-like): Covariance matrix of noise in target logits (default identity).
    - f (callable): Function to compute target logits from concepts and leakage (default MLP).
    - seed (int): Random seed for replicability.

    Returns:
    - X (ndarray): Features matrix of shape (n, d).
    - c (ndarray): Ground truth concepts matrix of shape (n, k).
    - c_hat (ndarray): Estimated concepts matrix of shape (n, k).
    - y (ndarray): Target labels array of shape (n,).
    """
    # Set random seed for replicability
    if seed is not None:
        np.random.seed(seed)


    # dimensionality warning
    if k > b:
        print("Warning: Dimensionality of the concept embedding (k) exceeds the number of features being projected (b). "
            "In this case, the data will effectively lie in a b-dimensional subspace within the k-dimensional space")
    if k > d - b - l:
        print("Warning: Dimensionality of the concept embedding (k) exceeds the number of features being projected (d-b-l). "
            "In this case, the data will effectively lie in a (d-b-l)-dimensional subspace within the k-dimensional space")

    # Ensure that sizes are appropriate
    if d < b:
        raise ValueError("Invalid parameter configuration: b has to be smaller (/equal to) d")
    

    # Set default values for mean and covariance matrices if not provided
    if mu_x is None:
        mu_x = np.zeros(d)
    if Sigma_x is None:
        Sigma_x = np.eye(d)
    if Sigma_c is None:
        Sigma_c = np.eye(k)
    if Sigma_c_hat is None:
        Sigma_c_hat = np.eye(k)
    if Sigma_y is None:
        Sigma_y = np.eye(J)

    # Generate features X ~ N(mu_x, Sigma_x)
    X = np.random.multivariate_normal(mu_x, Sigma_x, size=n)  # Shape: (n, d)

    # Construct matrix A for ground truth concepts
    R_A = np.random.normal(0, 1, size=(k, b))  # Shape: (k, b)
    A = np.hstack([R_A, np.zeros((k, d - b))])  # Shape: (k, d)

    # Generate noise for ground truth concepts
    epsilon_c = np.random.multivariate_normal(np.zeros(k), Sigma_c, size=n)  # Shape: (n, k)

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
    B[:, b : b + size_R_B] = R_B  # Fill in the appropriate columns

    # Compute leakage term l_i = B x_i
    l_i = np.dot(X, B.T)  # Shape: (n, k)

    # Generate noise for estimated concepts
    epsilon_c_hat = np.random.multivariate_normal(np.zeros(k), Sigma_c_hat, size=n)  # Shape: (n, k)

    # Compute logits for estimated concepts c_hat_i
    logits_c_hat = np.dot(X, A.T) + l_i + epsilon_c_hat  # Shape: (n, k)

    # Compute estimated concepts c_hat_i using the sigmoid function
    c_hat = sigmoid(logits_c_hat)  # Shape: (n, k)

    # Generate noise for target logits
    epsilon_y = np.random.multivariate_normal(np.zeros(J), Sigma_y, size=n)  # Shape: (n, J)

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
    logits_p = f(c, l_i) + epsilon_y  # Shape: (n, J)

    # Compute target probabilities using the softmax function
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
        return e_x / e_x.sum(axis=1, keepdims=True)

    p_i = softmax(logits_p)  # Shape: (n, J)

    # Sample target labels y_i ~ Categorical(p_i)
    y = np.array([np.random.choice(J, p=p_i[i]) for i in range(n)]) + 1  # Labels from 1 to J

    return X, c, c_hat, y

# EXAMPLE:

# # Generate synthetic data
# n = 1000   # Number of observations
# d = 50     # Number of features
# k = 10     # Number of concepts
# J = 5      # Number of target classes
# b = 20     # Number of features used in ground truth concepts (ensure k < b < d - k - l)
# l = 10     # Number of features excluded from leakage (ensure k < d - b - l)

# # Generate data
# X, c, c_hat, y = generate_synthetic_data_leakage(n, d, k, J, b, l, seed=42)