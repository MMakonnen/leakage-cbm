import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


# Function creating data splits in desired format
def split_data(X, c, c_hat, y_zero_based, train_size, val_size, test_size, seed):
    # First, split off the test set
    X_temp, X_test, c_temp, c_test, c_hat_temp, c_hat_test, y_temp, y_test = train_test_split(
        X, c, c_hat, y_zero_based, test_size=test_size, random_state=seed
    )

    # Then, split the remaining data into training and validation sets
    val_ratio = val_size / (train_size + val_size)  # Adjust validation size accordingly
    (
        X_train,
        X_val,
        c_train,
        c_val,
        c_hat_train,
        c_hat_val,
        y_train,
        y_val,
    ) = train_test_split(
        X_temp,
        c_temp,
        c_hat_temp,
        y_temp,
        test_size=val_ratio,
        random_state=seed,
    )
    # Now, training, validation, and test sets are split according to specified sizes
    return (
        X_train,
        X_val,
        X_test,
        c_train,
        c_val,
        c_test,
        c_hat_train,
        c_hat_val,
        c_hat_test,
        y_train,
        y_val,
        y_test,
    )


# ...
def prepare_data_loaders(
    c_train,
    c_val,
    c_test,
    c_hat_train,
    c_hat_val,
    c_hat_test,
    y_train,
    y_val,
    y_test,
    batch_size,
    seed,
    device,
):
    # Convert data to PyTorch tensors and move to the specified device
    c_train_tensor = torch.FloatTensor(c_train).to(device)
    c_val_tensor = torch.FloatTensor(c_val).to(device)
    c_test_tensor = torch.FloatTensor(c_test).to(device)

    c_hat_train_tensor = torch.FloatTensor(c_hat_train).to(device)
    c_hat_val_tensor = torch.FloatTensor(c_hat_val).to(device)
    c_hat_test_tensor = torch.FloatTensor(c_hat_test).to(device)

    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    # Create DataLoader for training, validation, and testing
    data_loader_seed = torch.Generator(device=device)
    data_loader_seed.manual_seed(seed)

    # Dataset and DataLoader for g_b (predicting y from c)
    train_dataset_gb = TensorDataset(c_train_tensor, y_train_tensor)
    val_dataset_gb = TensorDataset(c_val_tensor, y_val_tensor)
    test_dataset_gb = TensorDataset(c_test_tensor, y_test_tensor)

    train_loader_gb = DataLoader(
        train_dataset_gb,
        batch_size=batch_size,
        shuffle=True,
        generator=data_loader_seed,
    )
    val_loader_gb = DataLoader(val_dataset_gb, batch_size=batch_size, shuffle=False)
    test_loader_gb = DataLoader(test_dataset_gb, batch_size=batch_size, shuffle=False)

    # Dataset and DataLoader for g_a (predicting y from c_hat and c)
    # Concatenate c_hat and c
    c_hat_c_train_tensor = torch.cat((c_hat_train_tensor, c_train_tensor), dim=1)
    c_hat_c_val_tensor = torch.cat((c_hat_val_tensor, c_val_tensor), dim=1)
    c_hat_c_test_tensor = torch.cat((c_hat_test_tensor, c_test_tensor), dim=1)

    train_dataset_ga = TensorDataset(c_hat_c_train_tensor, y_train_tensor)
    val_dataset_ga = TensorDataset(c_hat_c_val_tensor, y_val_tensor)
    test_dataset_ga = TensorDataset(c_hat_c_test_tensor, y_test_tensor)

    train_loader_ga = DataLoader(
        train_dataset_ga,
        batch_size=batch_size,
        shuffle=True,
        generator=data_loader_seed,
    )
    val_loader_ga = DataLoader(val_dataset_ga, batch_size=batch_size, shuffle=False)
    test_loader_ga = DataLoader(test_dataset_ga, batch_size=batch_size, shuffle=False)

    return (
        train_loader_gb,
        val_loader_gb,
        test_loader_gb,
        train_loader_ga,
        val_loader_ga,
        test_loader_ga,
    )