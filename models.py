import torch.nn as nn


# simple MLP model
# NOTE: returns logits and NOT class probabilities or binary predicted target vec
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)  # Returns logits