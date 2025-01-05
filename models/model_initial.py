import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

from models.simple_mlp import SimpleMLP
from model_wrappers.model_wrappers import (
    PyTorchModelWrapper,
    SklearnModelWrapper,
    LightGBMModelWrapper,
    XGBoostModelWrapper
)

def initialize_models(c_train, c_hat_train, J, seed, device, model_type="mlp"):
    """
    Initializes and returns model wrappers and loss criterion based on the model type.

    Parameters:
    - c_train, c_hat_train: Training tensors for ground-truth and estimated concepts.
    - J: Number of classes.
    - seed: Random seed for reproducibility.
    - device: Device for computations ('cpu' or 'cuda').
    - model_type: Type of model to initialize ('mlp', 'random_forest', 'lightgbm', 'xgboost').

    Returns:
    - model_gb_wrapper, model_ga_wrapper: Wrappers for ground-truth and concatenated models.
    - criterion: Loss criterion (for training neural networks or consistency).
    """
    torch.manual_seed(seed)
    input_dim_gb = c_train.shape[1]  # Ground-truth concept dimension
    input_dim_ga = c_train.shape[1] + c_hat_train.shape[1]  # Concatenated dimension

    if model_type == "mlp":
        # Initialize MLP models
        model_gb = SimpleMLP(input_dim=input_dim_gb, num_classes=J).to(device)
        model_ga = SimpleMLP(input_dim=input_dim_ga, num_classes=J).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer_gb = optim.Adam(model_gb.parameters(), lr=0.001)
        optimizer_ga = optim.Adam(model_ga.parameters(), lr=0.001)
        model_gb_wrapper = PyTorchModelWrapper(model_gb, criterion, optimizer_gb, device)
        model_ga_wrapper = PyTorchModelWrapper(model_ga, criterion, optimizer_ga, device)
        return model_gb_wrapper, model_ga_wrapper, criterion

    elif model_type == "random_forest":
        # Initialize Random Forest models
        rf_gb = RandomForestClassifier(n_estimators=100, random_state=seed)
        rf_ga = RandomForestClassifier(n_estimators=100, random_state=seed)
        model_gb_wrapper = SklearnModelWrapper(rf_gb, device, n_classes=J)
        model_ga_wrapper = SklearnModelWrapper(rf_ga, device, n_classes=J)
        criterion = nn.CrossEntropyLoss().to(device)
        return model_gb_wrapper, model_ga_wrapper, criterion

    elif model_type == "lightgbm":
        # Initialize LightGBM models
        gbm_gb = lgb.LGBMClassifier(
            num_class=J,
            random_state=seed,
            min_data_in_leaf=10,
            min_split_gain=0.01,
            max_depth=15
        )
        gbm_ga = lgb.LGBMClassifier(
            num_class=J,
            random_state=seed,
            min_data_in_leaf=10,
            min_split_gain=0.01,
            max_depth=15
        )
        model_gb_wrapper = LightGBMModelWrapper(gbm_gb, device, n_classes=J)
        model_ga_wrapper = LightGBMModelWrapper(gbm_ga, device, n_classes=J)
        criterion = nn.CrossEntropyLoss().to(device)
        return model_gb_wrapper, model_ga_wrapper, criterion

    elif model_type == "xgboost":
        # Initialize XGBoost models
        gbm_gb = xgb.XGBClassifier(
            num_class=J,
            eval_metric='mlogloss',
            random_state=seed
        )
        gbm_ga = xgb.XGBClassifier(
            num_class=J,
            eval_metric='mlogloss',
            random_state=seed
        )
        model_gb_wrapper = XGBoostModelWrapper(gbm_gb, device, n_classes=J)
        model_ga_wrapper = XGBoostModelWrapper(gbm_ga, device, n_classes=J)
        criterion = nn.CrossEntropyLoss().to(device)  # For consistency
        return model_gb_wrapper, model_ga_wrapper, criterion

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
