from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
import numpy as np

from synthetic_dataset import generate_synthetic_data_leakage
from models import SimpleMLP
from utils import train_model, evaluate_model
from calibration import TemperatureScaler


###################
# parameters
###################

# ensure reproducibility (set random seeds)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


# synthetic data parameters
n = 1000   # Number of observations
d = 50     # Number of features
k = 10     # Number of concepts
J = 5      # Number of target classes
b = 20     # Number of features used in ground truth concepts (ensure k < b < d - k - l)
l = 10     # Number of features excluded from leakage (ensure k < d - b - l)

# DL model training parameters
batch_size = 64
num_epochs = 20

###################



# Generate synthetic data
X, c, c_hat, y = generate_synthetic_data_leakage(n, d, k, J, b, l, seed=seed)

# Convert labels to zero-based indexing for PyTorch
y_zero_based = y - 1


# Split data into training, validation (calibration), and test sets
X_temp, X_test, c_temp, c_test, c_hat_temp, c_hat_test, y_temp, y_test = train_test_split(
    X, c, c_hat, y_zero_based, test_size=0.2, random_state=42
)

X_train, X_val, c_train, c_val, c_hat_train, c_hat_val, y_train, y_val = train_test_split(
    X_temp, c_temp, c_hat_temp, y_temp, test_size=0.25, random_state=42
)
# Now, training data is 60%, validation (calibration) data is 20%, test data is 20%


# Convert data to PyTorch tensors (and move on device for accelerated computations)
# NOTE: CURRENTLY RUNNING ON CPU ONLY SINCE MPS THROWS FLOAT ERROR -> FIX THIS
device = torch.device('cpu')
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

c_train_tensor = torch.FloatTensor(c_train).to(device)
c_val_tensor = torch.FloatTensor(c_val).to(device)
c_test_tensor = torch.FloatTensor(c_test).to(device)

c_hat_train_tensor = torch.FloatTensor(c_hat_train).to(device)
c_hat_val_tensor = torch.FloatTensor(c_hat_val).to(device)
c_hat_test_tensor = torch.FloatTensor(c_hat_test).to(device)

y_train_tensor = torch.LongTensor(y_train).to(device)
y_val_tensor = torch.LongTensor(y_val).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)


# Create DataLoader for training, validation (calibration), and testing
data_loader_seed = torch.Generator()
data_loader_seed.manual_seed(seed)

# Dataset and DataLoader for g_b (predicting y from c)
train_dataset_gb = TensorDataset(c_train_tensor, y_train_tensor)
val_dataset_gb = TensorDataset(c_val_tensor, y_val_tensor)
test_dataset_gb = TensorDataset(c_test_tensor, y_test_tensor)

train_loader_gb = DataLoader(train_dataset_gb, batch_size=batch_size, shuffle=True, generator=data_loader_seed)
val_loader_gb = DataLoader(val_dataset_gb, batch_size=batch_size, shuffle=False)
test_loader_gb = DataLoader(test_dataset_gb, batch_size=batch_size, shuffle=False)

# Dataset and DataLoader for g_a (predicting y from c_hat and c)
# Concatenate c_hat and c (new dim is dim(c_hat)+dim(c))
c_hat_c_train_tensor = torch.cat((c_hat_train_tensor, c_train_tensor), dim=1)
c_hat_c_val_tensor = torch.cat((c_hat_val_tensor, c_val_tensor), dim=1)
c_hat_c_test_tensor = torch.cat((c_hat_test_tensor, c_test_tensor), dim=1)

train_dataset_ga = TensorDataset(c_hat_c_train_tensor, y_train_tensor)
val_dataset_ga = TensorDataset(c_hat_c_val_tensor, y_val_tensor)
test_dataset_ga = TensorDataset(c_hat_c_test_tensor, y_test_tensor)

train_loader_ga = DataLoader(train_dataset_ga, batch_size=batch_size, shuffle=True, generator=data_loader_seed)
val_loader_ga = DataLoader(val_dataset_ga, batch_size=batch_size, shuffle=False)
test_loader_ga = DataLoader(test_dataset_ga, batch_size=batch_size, shuffle=False)


# Initialize models, loss function, and optimizer
input_dim_gb = c_train_tensor.shape[1]
input_dim_ga = c_hat_c_train_tensor.shape[1]
num_classes = J  # Number of classes

torch.manual_seed(seed)
model_gb = SimpleMLP(input_dim=input_dim_gb, num_classes=num_classes).to(device)
model_ga = SimpleMLP(input_dim=input_dim_ga, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()

optimizer_gb = optim.Adam(model_gb.parameters(), lr=0.001)
optimizer_ga = optim.Adam(model_ga.parameters(), lr=0.001)


# Training loop
print("Training g_b (predicting y from c)...")
for epoch in range(num_epochs):
    train_loss, train_acc = train_model(model_gb, optimizer_gb, criterion, train_loader_gb)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

print("\nTraining g_a (predicting y from c_hat and c)...")
for epoch in range(num_epochs):
    train_loss, train_acc = train_model(model_ga, optimizer_ga, criterion, train_loader_ga)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")


# Calibrate models using validation data
print("\nCalibrating g_b using validation data...")
temperature_scaler_gb = TemperatureScaler().to(device)
# Collect logits and labels from validation set
model_gb.eval()
with torch.no_grad():
    logits_list = []
    labels_list = []
    for inputs, labels in val_loader_gb:
        logits = model_gb(inputs)
        logits_list.append(logits)
        labels_list.append(labels)
    val_logits_gb = torch.cat(logits_list)
    val_labels_gb = torch.cat(labels_list)
# Optimize temperature
temperature_scaler_gb.set_temperature(val_logits_gb, val_labels_gb)
print(f"Optimal temperature for g_b: {temperature_scaler_gb.temperature.item():.4f}")

print("\nCalibrating g_a using validation data...")
temperature_scaler_ga = TemperatureScaler().to(device)
# Collect logits and labels from validation set
model_ga.eval()
with torch.no_grad():
    logits_list = []
    labels_list = []
    for inputs, labels in val_loader_ga:
        logits = model_ga(inputs)
        logits_list.append(logits)
        labels_list.append(labels)
    val_logits_ga = torch.cat(logits_list)
    val_labels_ga = torch.cat(labels_list)
# Optimize temperature
temperature_scaler_ga.set_temperature(val_logits_ga, val_labels_ga)
print(f"Optimal temperature for g_a: {temperature_scaler_ga.temperature.item():.4f}")


# Evaluate models and compute negative log-likelihoods
print("\nEvaluating g_b on test data...")
test_loss_gb, test_acc_gb, avg_nll_gb = evaluate_model(
    model_gb, temperature_scaler_gb, criterion, test_loader_gb
)
print(f"Test Loss: {test_loss_gb:.4f}, Accuracy: {test_acc_gb:.4f}, Avg NLL: {avg_nll_gb:.4f}")

print("\nEvaluating g_a on test data...")
test_loss_ga, test_acc_ga, avg_nll_ga = evaluate_model(
    model_ga, temperature_scaler_ga, criterion, test_loader_ga
)
print(f"Test Loss: {test_loss_ga:.4f}, Accuracy: {test_acc_ga:.4f}, Avg NLL: {avg_nll_ga:.4f}")

# Compute the estimated entropies
H_y_given_c = avg_nll_gb  # H(y | c)
H_y_given_z_c = avg_nll_ga  # H(y | z, c)

# Compute the leakage measure
leakage_estimate = H_y_given_c - H_y_given_z_c

print(f"\nEstimated H(y | c): {H_y_given_c:.4f}")
print(f"Estimated H(y | z, c): {H_y_given_z_c:.4f}")
print(f"Estimated Leakage I(z; y | c): {leakage_estimate:.4f}")