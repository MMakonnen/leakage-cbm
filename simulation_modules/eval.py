import numpy as np
import torch

# Evaluate the model on a dataset
def evaluate_model(model_wrapper, data_loader, criterion, device):
    """
    Evaluates the model to compute loss, accuracy, and average NLL.

    Parameters:
    - model_wrapper: Provides model methods like predict_proba.
    - data_loader: DataLoader with inputs and true labels.
    - criterion: Loss function (e.g., CrossEntropyLoss).
    - device: Device for computation ('cpu' or 'cuda').

    Returns:
    - loss: Computed loss on the dataset.
    - acc: Prediction accuracy.
    - avg_nll: Average negative log-likelihood.
    """
    # Predict probabilities
    probs = model_wrapper.predict_proba(data_loader)
    
    # Extract true labels from the DataLoader
    labels_list = [labels.cpu().numpy() for _, labels in data_loader]
    y_true = np.concatenate(labels_list)
    
    # Get predicted labels
    y_pred = np.argmax(probs, axis=1)
    
    # Prepare logits and labels for loss calculation
    logits_for_loss = torch.tensor(np.log(np.clip(probs, 1e-9, 1.0)), dtype=torch.float32, device=device)
    labels_for_loss = torch.tensor(y_true, dtype=torch.long, device=device)
    
    # Compute loss and accuracy
    loss = criterion(logits_for_loss, labels_for_loss).item()
    acc = np.mean(y_pred == y_true)
    avg_nll = loss  # Average NLL is equivalent to the loss
    
    return loss, acc, avg_nll

# Aggregate results from multiple simulations
def average_results(simulation_results):
    """
    Computes average metrics across simulation results.

    Parameters:
    - simulation_results: List of dictionaries with simulation metrics.

    Returns:
    - averaged_result: Dictionary with averaged metrics and shared parameters.
    """
    # Extract shared parameters
    param_keys = [
        'n', 'd', 'k', 'J', 'b', 'l',
        'batch_size', 'num_epochs',
        'train_size', 'val_size', 'test_size',
        'num_simulations', 'model_type'
    ]
    params = {k: simulation_results[0][k] for k in param_keys if k in simulation_results[0]}
    
    # Compute averages for metrics
    metrics_keys = [k for k in simulation_results[0] if k not in params]
    averaged_metrics = {key: np.mean([res[key] for res in simulation_results]) for key in metrics_keys}
    
    # Combine parameters and averaged metrics
    averaged_result = {**params, **averaged_metrics}
    
    return averaged_result
