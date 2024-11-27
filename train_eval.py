import torch
import numpy as np

# Function to train DL model
def train_model(model, optimizer, criterion, data_loader, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    for inputs, labels in data_loader:
        # Ensure inputs and labels are on the correct device
        inputs = inputs.to(device)
        labels = labels.to(device)

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


def train_model_generic(model, optimizer, criterion, train_loader, num_epochs, device):
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, optimizer, criterion, train_loader, device)
        # Optionally, print progress
        # if (epoch + 1) % 5 == 0 or epoch == 0:
        #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")


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