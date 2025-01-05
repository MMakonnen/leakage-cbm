import torch
import torch.nn as nn
import torch.optim as optim

# Module for temperature scaling, calibrating model confidence
class TemperatureScaler(nn.Module):
    def __init__(self):
        """
        Initialize the TemperatureScaler module.

        - The module contains a learnable parameter `temperature`, initialized to 1.0.
        """
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)  # Learnable temperature parameter

    def forward(self, logits):
        """
        Scale logits by dividing by the temperature.

        Parameters:
        - logits: Tensor of raw outputs from the model.

        Returns:
        - Scaled logits.
        """
        return logits / self.temperature

    def set_temperature(self, logits, labels):
        """
        Optimize the temperature parameter to minimize negative log-likelihood (NLL).

        Parameters:
        - logits: Model outputs (before softmax).
        - labels: Ground truth labels.
        """
        nll_criterion = nn.CrossEntropyLoss()  # Loss function for classification
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        # Detach logits and labels to ensure no gradients are propagated to the model
        logits = logits.detach()
        labels = labels.detach()
        
        # Closure function for LBFGS optimization
        def eval():
            optimizer.zero_grad()  # Clear gradients
            loss = nll_criterion(logits / self.temperature, labels)  # Compute NLL with scaled logits
            loss.backward()  # Backpropagate to compute gradients
            return loss

        optimizer.step(eval)  # Optimize temperature parameter
