from synthetic_dataset import generate_synthetic_data_leakage
from sklearn.model_selection import train_test_split

seed = 42

# Generate synthetic data
n = 1000   # Number of observations
d = 50     # Number of features
k = 10     # Number of concepts
J = 5      # Number of target classes
b = 20     # Number of features used in ground truth concepts (ensure k < b < d - k - l)
l = 10     # Number of features excluded from leakage (ensure k < d - b - l)

# Generate data
X, c, c_hat, y = generate_synthetic_data_leakage(n, d, k, J, b, l, seed=seed)

# Convert labels to zero-based indexing for PyTorch
y_zero_based = y - 1


# Split data into training, validation (calibration), and test sets
X_temp, X_test, c_temp, c_test, hat_c_temp, hat_c_test, y_temp, y_test = train_test_split(
    X, c, c_hat, y_zero_based, test_size=0.2, random_state=42
)

X_train, X_val, c_train, c_val, hat_c_train, hat_c_val, y_train, y_val = train_test_split(
    X_temp, c_temp, hat_c_temp, y_temp, test_size=0.25, random_state=42
)
# Now, training data is 60%, validation (calibration) data is 20%, test data is 20%