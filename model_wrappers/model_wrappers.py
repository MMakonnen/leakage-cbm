import torch
import numpy as np
from calibration.calibration import TemperatureScaler  # Import for model calibration

# Base class defining a common interface for models
class BaseModelWrapper:
    def fit(self, train_loader, num_epochs=None):
        raise NotImplementedError("Subclasses must implement this method.")

    def predict_proba(self, data_loader):
        raise NotImplementedError("Subclasses must implement this method.")

    def calibrate(self, val_loader):
        # Default: no calibration; can be overridden in subclasses
        pass


# Wrapper for PyTorch models
class PyTorchModelWrapper(BaseModelWrapper):
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.temperature_scaler = None

    def fit(self, train_loader, num_epochs=20):
        """
        Trains the PyTorch model.
        """
        self.model.train()
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

    def predict_proba(self, data_loader):
        """
        Predicts probabilities using the model.
        """
        self.model.eval()
        if self.temperature_scaler:
            self.temperature_scaler.eval()
        all_probs = []
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                logits = self.model(inputs)
                if self.temperature_scaler:
                    logits = self.temperature_scaler(logits)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.vstack(all_probs)

    def calibrate(self, val_loader):
        """
        Calibrates the model using temperature scaling.
        """
        self.temperature_scaler = TemperatureScaler().to(self.device)
        self.model.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)
        val_logits = torch.cat(logits_list)
        val_labels = torch.cat(labels_list)
        self.temperature_scaler.set_temperature(val_logits, val_labels)


# Wrapper for Scikit-learn models
class SklearnModelWrapper(BaseModelWrapper):
    def __init__(self, model, device, n_classes):
        self.model = model
        self.device = device
        self.n_classes = n_classes
        self.temperature_scaler = None

    def fit(self, train_loader, num_epochs=None):
        """
        Trains the Scikit-learn model.
        """
        X_list, y_list = [], []
        for inputs, labels in train_loader:
            X_list.append(inputs.cpu().numpy())
            y_list.append(labels.cpu().numpy())
        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)
        self.model.fit(X_train, y_train)

    def predict_proba(self, data_loader):
        """
        Predicts probabilities using the model.
        """
        X_list = []
        for inputs, _ in data_loader:
            X_list.append(inputs.cpu().numpy())
        X_data = np.vstack(X_list)
        probs = self.model.predict_proba(X_data)

        # Apply temperature scaling if calibrated
        if self.temperature_scaler:
            eps = 1e-9
            probs_clamped = np.clip(probs, eps, 1 - eps)
            log_probs = np.log(probs_clamped)
            T = self.temperature_scaler.temperature.detach().cpu().item()
            scaled_log_probs = log_probs / T
            max_log = np.max(scaled_log_probs, axis=1, keepdims=True)
            exp_z = np.exp(scaled_log_probs - max_log)
            scaled_probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            return scaled_probs

        return probs

    def calibrate(self, val_loader):
        """
        Calibrates the model using temperature scaling.
        """
        X_list, y_list = [], []
        for inputs, labels in val_loader:
            X_list.append(inputs.cpu().numpy())
            y_list.append(labels.cpu().numpy())
        X_val = np.vstack(X_list)
        y_val = np.concatenate(y_list)

        probs = self.model.predict_proba(X_val)
        eps = 1e-9
        probs_clamped = np.clip(probs, eps, 1 - eps)
        log_probs = np.log(probs_clamped)

        self.temperature_scaler = TemperatureScaler().to(self.device)
        logits_tensor = torch.tensor(log_probs, dtype=torch.float32, device=self.device)
        labels_tensor = torch.tensor(y_val, dtype=torch.long, device=self.device)
        self.temperature_scaler.set_temperature(logits_tensor, labels_tensor)


# Wrapper for LightGBM models
class LightGBMModelWrapper(SklearnModelWrapper):
    # No additional functionality required for now
    pass


# Wrapper for XGBoost models
class XGBoostModelWrapper(SklearnModelWrapper):
    def __init__(self, model, device, n_classes):
        super().__init__(model, device, n_classes)
        # XGBoost uses 'num_class' parameter instead of 'n_classes'
        if hasattr(self.model, 'set_params'):
            self.model.set_params(num_class=n_classes)
        else:
            raise AttributeError("The provided XGBoost model does not have a 'set_params' method.")
