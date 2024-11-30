import torch
import torch.nn as nn
import torch.optim as optim

# Temperature scaling class
class TemperatureScaler(nn.Module):
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
    
    def forward(self, logits):
        # During evaluation, we divide the logits by the temperature
        return logits / self.temperature
    
    def set_temperature(self, logits, labels):
        # Optimize the temperature using NLL on validation data
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        logits = logits.detach()
        labels = labels.detach()
        
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval)


# ...
def calibrate_model(model, val_loader, criterion, device):
    temperature_scaler = TemperatureScaler().to(device)
    # Collect logits and labels from validation set
    model.eval()
    with torch.no_grad():
        logits_list = []
        labels_list = []
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            logits_list.append(logits)
            labels_list.append(labels)
        val_logits = torch.cat(logits_list)
        val_labels = torch.cat(labels_list)
    # Optimize temperature
    temperature_scaler.set_temperature(val_logits, val_labels)
    return temperature_scaler