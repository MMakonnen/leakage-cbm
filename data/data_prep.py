from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

# Splits data into train, validation, and test sets
def split_data(X, c, c_hat, y_zero_based, train_size, val_size, test_size, seed):
    """
    Splits data into train, validation, and test sets.

    Parameters:
    - X, c, c_hat, y_zero_based: Input features and labels.
    - train_size, val_size, test_size: Proportions for splits.
    - seed: Random state for reproducibility.

    Returns:
    - Split data for X, c, c_hat, and labels.
    """
    # First split: test set vs. train + validation
    X_temp, X_test, c_temp, c_test, c_hat_temp, c_hat_test, y_temp, y_test = train_test_split(
        X, c, c_hat, y_zero_based, test_size=test_size, random_state=seed
    )

    # Second split: train vs. validation
    val_ratio = val_size / (train_size + val_size)
    X_train, X_val, c_train, c_val, c_hat_train, c_hat_val, y_train, y_val = train_test_split(
        X_temp, c_temp, c_hat_temp, y_temp, test_size=val_ratio, random_state=seed
    )

    return (
        X_train, X_val, X_test,
        c_train, c_val, c_test,
        c_hat_train, c_hat_val, c_hat_test,
        y_train, y_val, y_test,
    )

# Prepares PyTorch DataLoader objects
def prepare_data_loaders(
    c_train, c_val, c_test,
    c_hat_train, c_hat_val, c_hat_test,
    y_train, y_val, y_test,
    batch_size, seed, device
):
    """
    Creates DataLoader objects for training, validation, and testing.

    Parameters:
    - c_train, c_val, c_test: Ground-truth concept tensors.
    - c_hat_train, c_hat_val, c_hat_test: Estimated concept tensors.
    - y_train, y_val, y_test: Label tensors.
    - batch_size: Batch size for DataLoader.
    - seed: Random seed for reproducibility.
    - device: Device for tensor allocation.

    Returns:
    - DataLoaders for ground-truth-based (gb) and concatenated (ga) datasets.
    """
    # Convert data to tensors
    c_train_tensor = torch.FloatTensor(c_train).to(device)
    c_val_tensor = torch.FloatTensor(c_val).to(device)
    c_test_tensor = torch.FloatTensor(c_test).to(device)

    c_hat_train_tensor = torch.FloatTensor(c_hat_train).to(device)
    c_hat_val_tensor = torch.FloatTensor(c_hat_val).to(device)
    c_hat_test_tensor = torch.FloatTensor(c_hat_test).to(device)

    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    # Set random seed for shuffling
    data_loader_seed = torch.Generator(device=device)
    data_loader_seed.manual_seed(seed)

    # Create DataLoaders for ground-truth-based data
    train_loader_gb = DataLoader(
        TensorDataset(c_train_tensor, y_train_tensor),
        batch_size=batch_size, shuffle=True, generator=data_loader_seed
    )
    val_loader_gb = DataLoader(
        TensorDataset(c_val_tensor, y_val_tensor),
        batch_size=batch_size, shuffle=False
    )
    test_loader_gb = DataLoader(
        TensorDataset(c_test_tensor, y_test_tensor),
        batch_size=batch_size, shuffle=False
    )

    # Create DataLoaders for concatenated data
    train_loader_ga = DataLoader(
        TensorDataset(torch.cat((c_hat_train_tensor, c_train_tensor), dim=1), y_train_tensor),
        batch_size=batch_size, shuffle=True, generator=data_loader_seed
    )
    val_loader_ga = DataLoader(
        TensorDataset(torch.cat((c_hat_val_tensor, c_val_tensor), dim=1), y_val_tensor),
        batch_size=batch_size, shuffle=False
    )
    test_loader_ga = DataLoader(
        TensorDataset(torch.cat((c_hat_test_tensor, c_test_tensor), dim=1), y_test_tensor),
        batch_size=batch_size, shuffle=False
    )

    return train_loader_gb, val_loader_gb, test_loader_gb, train_loader_ga, val_loader_ga, test_loader_ga
