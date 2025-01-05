import torch.nn as nn

class SimpleMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) with one hidden layer.

    Parameters:
    - input_dim: Number of input features.
    - num_classes: Number of output classes.
    - hidden_dim: Number of units in the hidden layer (default: 64).
    """
    def __init__(self, input_dim, num_classes, hidden_dim=64):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the MLP.

        Parameters:
        - x: Input tensor.

        Returns:
        - Output tensor after passing through the model.
        """
        return self.model(x)
