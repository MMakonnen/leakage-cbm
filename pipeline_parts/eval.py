import numpy as np
import torch

def evaluate_model(model_wrapper, data_loader, criterion, device):
    probs = model_wrapper.predict_proba(data_loader)
    labels_list = []
    for _, labels in data_loader:
        labels_list.append(labels.cpu().numpy())
    y_true = np.concatenate(labels_list)
    y_pred = np.argmax(probs, axis=1)
    logits_for_loss = torch.tensor(np.log(np.clip(probs, 1e-9, 1.0)), dtype=torch.float32, device=device)
    labels_for_loss = torch.tensor(y_true, dtype=torch.long, device=device)
    loss = criterion(logits_for_loss, labels_for_loss).item()
    acc = np.mean(y_pred == y_true)
    avg_nll = loss
    return loss, acc, avg_nll

def average_results(simulation_results):
    param_keys = [
        'n', 'd', 'k', 'J', 'b', 'l',
        'batch_size', 'num_epochs',
        'train_size', 'val_size', 'test_size',
        'num_simulations', 'model_type'
    ]
    params = {k: simulation_results[0][k] for k in param_keys if k in simulation_results[0]}
    metrics_keys = [k for k in simulation_results[0] if k not in params]
    averaged_metrics = {}
    for key in metrics_keys:
        values = [res[key] for res in simulation_results]
        averaged_metrics[key] = np.mean(values)
    averaged_result = params.copy()
    averaged_result.update(averaged_metrics)
    return averaged_result
