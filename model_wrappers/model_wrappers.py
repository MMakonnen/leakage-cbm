import torch
import numpy as np

# Import the TemperatureScaler from calibration.py
from calibration.calibration import TemperatureScaler



class BaseModelWrapper:
    def fit(self, train_loader, num_epochs=None):
        raise NotImplementedError("Subclasses must implement this method.")

    def predict_proba(self, data_loader):
        raise NotImplementedError("Subclasses must implement this method.")

    def calibrate(self, val_loader):
        # By default, do nothing. Subclasses can override this if they need calibration.
        pass


class PyTorchModelWrapper(BaseModelWrapper):
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.temperature_scaler = None

    def fit(self, train_loader, num_epochs=20):
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


class SklearnModelWrapper(BaseModelWrapper):
    def __init__(self, model, device, n_classes):
        self.model = model
        self.device = device
        self.n_classes = n_classes
        self.temperature_scaler = None

    def fit(self, train_loader, num_epochs=None):
        X_list, y_list = [], []
        for inputs, labels in train_loader:
            X_list.append(inputs.cpu().numpy())
            y_list.append(labels.cpu().numpy())
        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)
        self.model.fit(X_train, y_train)

    def predict_proba(self, data_loader):
        X_list = []
        for inputs, _ in data_loader:
            X_list.append(inputs.cpu().numpy())
        X_data = np.vstack(X_list)
        probs = self.model.predict_proba(X_data)

        # If temperature scaling is applied, adjust probabilities
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
        X_list, y_list = [], []
        for inputs, labels in val_loader:
            X_list.append(inputs.cpu().numpy())
            y_list.append(labels.cpu().numpy())
        X_val = np.vstack(X_list)
        y_val = np.concatenate(y_list)

        probs = self.model.predict_proba(X_val)

        # Convert probabilities to "logits" for temperature scaling
        eps = 1e-9
        probs_clamped = np.clip(probs, eps, 1 - eps)
        log_probs = np.log(probs_clamped)

        self.temperature_scaler = TemperatureScaler().to(self.device)
        logits_tensor = torch.tensor(log_probs, dtype=torch.float32, device=self.device)
        labels_tensor = torch.tensor(y_val, dtype=torch.long, device=self.device)
        self.temperature_scaler.set_temperature(logits_tensor, labels_tensor)


class LightGBMModelWrapper(SklearnModelWrapper):
    # Inherits from SklearnModelWrapper because LightGBM's LGBMClassifier
    # has a similar API (predict_proba). No changes needed unless you want custom behavior.
    pass
