import numpy as np
import torch

# Evaluate the model on a given dataset
def evaluate_model(model_wrapper, data_loader, criterion, device):
    """
    Computes loss, accuracy, and average negative log-likelihood (NLL) for a model.

    Parameters:
    - model_wrapper: Wrapper object providing model interface (e.g., predict_proba).
    - data_loader: DataLoader containing inputs and true labels.
    - criterion: Loss function (e.g., CrossEntropyLoss).
    - device: Device to perform computations ('cpu' or 'cuda').

    Returns:
    - loss: Computed loss on the dataset.
    - acc: Accuracy of predictions.
    - avg_nll: Average negative log-likelihood.
    """
    # Predict probabilities using the model wrapper
    probs = model_wrapper.predict_proba(data_loader)
    
    # Extract true labels from the data loader
    labels_list = []
    for _, labels in data_loader:
        labels_list.append(labels.cpu().numpy())
    y_true = np.concatenate(labels_list)
    
    # Determine predicted labels
    y_pred = np.argmax(probs, axis=1)
    
    # Prepare logits and labels for loss calculation
    logits_for_loss = torch.tensor(np.log(np.clip(probs, 1e-9, 1.0)), dtype=torch.float32, device=device)
    labels_for_loss = torch.tensor(y_true, dtype=torch.long, device=device)
    
    # Compute loss and accuracy
    loss = criterion(logits_for_loss, labels_for_loss).item()
    acc = np.mean(y_pred == y_true)
    
    # Average negative log-likelihood is equivalent to the loss
    avg_nll = loss
    
    return loss, acc, avg_nll

# Average results from multiple simulations
def average_results(simulation_results):
    """
    Aggregates simulation results by computing average metrics across simulations.

    Parameters:
    - simulation_results: List of dictionaries containing parameters and metrics for each simulation.

    Returns:
    - averaged_result: Dictionary with averaged metrics and shared parameters.
    """
    # Extract shared parameters from the first simulation result
    param_keys = [
        'n', 'd', 'k', 'J', 'b', 'l',
        'batch_size', 'num_epochs',
        'train_size', 'val_size', 'test_size',
        'num_simulations', 'model_type'
    ]
    params = {k: simulation_results[0][k] for k in param_keys if k in simulation_results[0]}
    
    # Identify metrics to average (keys not in the parameter list)
    metrics_keys = [k for k in simulation_results[0] if k not in params]
    
    # Compute averages for all metrics
    averaged_metrics = {key: np.mean([res[key] for res in simulation_results]) for key in metrics_keys}
    
    # Combine parameters and averaged metrics
    averaged_result = params.copy()
    averaged_result.update(averaged_metrics)
    
    return averaged_result
