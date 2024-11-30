import torch
import torch.nn as nn
import torch.optim as optim


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
    

def initialize_mlps(c_train, c_hat_train, J, seed, device):
    torch.manual_seed(seed)
    input_dim_gb = c_train.shape[1]
    input_dim_ga = c_train.shape[1] + c_hat_train.shape[1]  # Concatenated input

    model_gb = SimpleMLP(input_dim=input_dim_gb, num_classes=J).to(device)
    model_ga = SimpleMLP(input_dim=input_dim_ga, num_classes=J).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer_gb = optim.Adam(model_gb.parameters(), lr=0.001)
    optimizer_ga = optim.Adam(model_ga.parameters(), lr=0.001)

    return model_gb, model_ga, optimizer_gb, optimizer_ga, criterion



# def initialize_models(c_train, c_hat_train, J, seed, device):
#     torch.manual_seed(seed)
#     input_dim_gb = c_train.shape[1]
#     input_dim_ga = c_train.shape[1] + c_hat_train.shape[1]  # Concatenated input

#     model_gb = SimpleMLP(input_dim=input_dim_gb, num_classes=J).to(device)
#     model_ga = SimpleMLP(input_dim=input_dim_ga, num_classes=J).to(device)

#     criterion = nn.CrossEntropyLoss().to(device)
#     optimizer_gb = optim.Adam(model_gb.parameters(), lr=0.001)
#     optimizer_ga = optim.Adam(model_ga.parameters(), lr=0.001)

#     return model_gb, model_ga, optimizer_gb, optimizer_ga, criterion