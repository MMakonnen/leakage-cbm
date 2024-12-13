import torch
import torch.nn as nn
import torch.optim as optim

# Module for temperature scaling, used to calibrate model confidence
class TemperatureScaler(nn.Module):
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        # Learnable parameter for temperature scaling, initialized to 1.0
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
    
    def forward(self, logits):
        # Scale the logits by dividing them by the temperature
        return logits / self.temperature

    def set_temperature(self, logits, labels):
        """
        Optimize the temperature parameter to minimize the negative log-likelihood (NLL)
        between scaled logits and ground truth labels.
        
        Parameters:
        - logits: Model's raw outputs (before softmax).
        - labels: Ground truth labels for the inputs.
        """
        nll_criterion = nn.CrossEntropyLoss()  # Loss function for classification
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        # Detach logits and labels to ensure no gradients are propagated to the model
        logits = logits.detach()
        labels = labels.detach()
        
        # Closure function optimizer
        def eval():
            optimizer.zero_grad()  # Clear gradients
            loss = nll_criterion(logits / self.temperature, labels)  # Compute NLL with scaled logits
            loss.backward()  # Backpropagate to compute gradients
            return loss

        # Optimize the temperature parameter
        optimizer.step(eval)
