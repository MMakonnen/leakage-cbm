import torch

# Function to train DL model
def train_model(model, optimizer, criterion, data_loader):
    model.train()
    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    for inputs, labels in data_loader:
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        _, preds = torch.max(logits, 1)
        correct_predictions += torch.sum(preds == labels.data)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    return epoch_loss, epoch_acc.item()


# Function to evaluate DL model and compute negative log-likelihoods
def evaluate_model(model, temperature_scaler, criterion, data_loader):
    model.eval()
    temperature_scaler.eval()
    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    negative_log_likelihoods = []

    with torch.no_grad():
        for inputs, labels in data_loader:
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