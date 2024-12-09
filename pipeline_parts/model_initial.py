import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from FINAL_CODE.models.simple_mlp import SimpleMLP
from FINAL_CODE.model_wrappers.model_wrappers import PyTorchModelWrapper, SklearnModelWrapper, LightGBMModelWrapper

def initialize_models(c_train, c_hat_train, J, seed, device, model_type="mlp"):
    torch.manual_seed(seed)
    input_dim_gb = c_train.shape[1]
    input_dim_ga = c_train.shape[1] + c_hat_train.shape[1]

    if model_type == "mlp":
        model_gb = SimpleMLP(input_dim=input_dim_gb, num_classes=J).to(device)
        model_ga = SimpleMLP(input_dim=input_dim_ga, num_classes=J).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer_gb = optim.Adam(model_gb.parameters(), lr=0.001)
        optimizer_ga = optim.Adam(model_ga.parameters(), lr=0.001)
        model_gb_wrapper = PyTorchModelWrapper(model_gb, criterion, optimizer_gb, device)
        model_ga_wrapper = PyTorchModelWrapper(model_ga, criterion, optimizer_ga, device)
        return model_gb_wrapper, model_ga_wrapper, criterion

    elif model_type == "random_forest":
        rf_gb = RandomForestClassifier(n_estimators=100, random_state=seed)
        rf_ga = RandomForestClassifier(n_estimators=100, random_state=seed)
        model_gb_wrapper = SklearnModelWrapper(rf_gb, device, n_classes=J)
        model_ga_wrapper = SklearnModelWrapper(rf_ga, device, n_classes=J)
        criterion = nn.CrossEntropyLoss().to(device)
        return model_gb_wrapper, model_ga_wrapper, criterion

    elif model_type == "xgboost":
        gbm_gb = lgb.LGBMClassifier(num_class=J, random_state=seed)
        gbm_ga = lgb.LGBMClassifier(num_class=J, random_state=seed)
        model_gb_wrapper = LightGBMModelWrapper(gbm_gb, device, n_classes=J)
        model_ga_wrapper = LightGBMModelWrapper(gbm_ga, device, n_classes=J)
        criterion = nn.CrossEntropyLoss().to(device)
        return model_gb_wrapper, model_ga_wrapper, criterion

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
