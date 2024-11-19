import torch
import torch.nn as nn

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