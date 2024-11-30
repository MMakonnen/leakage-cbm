import torch
import torch.nn as nn
import numpy as np

# Function to evaluate DL model and compute negative log-likelihoods
def evaluate_model(model, temperature_scaler, criterion, data_loader, device):
    model.eval()
    temperature_scaler.eval()
    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    negative_log_likelihoods = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            # Ensure inputs and labels are on the correct device
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            # Apply temperature scaling
            scaled_logits = temperature_scaler(logits)
            # Compute probabilities
            probabilities = nn.functional.softmax(scaled_logits, dim=1)
            # Compute negative log-likelihoods
            nll = -torch.log(probabilities.gather(1, labels.unsqueeze(1)))
            negative_log_likelihoods.extend(nll.squeeze().tolist())
            # Compute loss
            loss = criterion(scaled_logits, labels)
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            # Predictions
            _, preds = torch.max(probabilities, 1)
            correct_predictions += torch.sum(preds == labels.data)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    avg_nll = np.mean(negative_log_likelihoods)

    return epoch_loss, epoch_acc.item(), avg_nll


# ...
def average_results(simulation_results):
    # Parameters that are consistent across runs (excluding 'seed')
    param_keys = [
        'n', 'd', 'k', 'J', 'b', 'l',
        'batch_size', 'num_epochs',
        'train_size', 'val_size', 'test_size',
        'num_simulations'
    ]
    params = {k: simulation_results[0][k] for k in param_keys}
    # Metrics to average
    metrics_keys = [k for k in simulation_results[0] if k not in params]
    averaged_metrics = {}
    for key in metrics_keys:
        values = [res[key] for res in simulation_results]
        averaged_metrics[key] = np.mean(values)
    # Combine parameters and averaged_metrics
    averaged_result = params.copy()
    averaged_result.update(averaged_metrics)
    return averaged_result