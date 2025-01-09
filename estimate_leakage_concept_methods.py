

# - autoregressive model not trained joined hence that is not entered properly and hence is also an issue with the visualizations
# - double check in plots of autoreg and embedding are even trained joined, correct these in the plots !!!



# - fix heatmap to be just squares for different values
# - move visualiuations to other folder
# run final simulations with better par: pick 100 or 50 or 200



# - once fixed saving store final simulation results and use those for analysis indep of what happens ...
# - crucially talk about how results tends to be relatively sensitive to parameters ... -> hence this requires furhter investigation
# - save file properly
# - write about this concisely and sicuss that probably requites more engineering and look int ofurther, more engineering at this very moment
# -> for example tuning default aplha seems to make big diff
# -> do more ablations usw for parameters (e.g. num, num par, ...)
# - split files in here up properly !!!
# - MAKE CLEAR THAT TOOK SIGNIFICANT CHUNK OF CODE FROM THE SCBM REPO

# FIX: num sim runs, maybe n, think about parameters overall without violating constraints, improve visualizations, 
# ensure no train sets that dont contain a class else skip to next, how can this happen, improve data gen seems weird with the
# amopunt of observations, does this have somethign todo with batching???
# reserach some more 




import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import xgboost as xgb
from sklearn.metrics import log_loss
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
from datetime import datetime

from data.synthetic_data_gen import generate_synthetic_data_leakage

# =========================================================
# NOTE: Significant portions of this code were adapted from the SCBM repository ...
# =========================================================

# =========================================================
# FREEZE / UNFREEZE UTILS
# =========================================================

def freeze_module(m):
    m.eval()
    for param in m.parameters():
        param.requires_grad = False

def unfreeze_module(m):
    m.train()
    for param in m.parameters():
        param.requires_grad = True

# =========================================================
# CBM IMPLEMENTATION
# =========================================================

class FCNNEncoder(nn.Module):
    """
    A simple feedforward encoder for the CBM.
    """
    def __init__(self, num_inputs: int, num_hidden: int, num_deep: int):
        super(FCNNEncoder, self).__init__()
        self.fc0 = nn.Linear(num_inputs, num_hidden)
        self.bn0 = nn.BatchNorm1d(num_hidden)
        self.fcs = nn.ModuleList(
            [nn.Linear(num_hidden, num_hidden) for _ in range(num_deep)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(num_deep)])
        self.dp = nn.Dropout(0.3)  # Increased dropout rate for better regularization

    def forward(self, x):
        z = self.bn0(self.dp(F.relu(self.fc0(x))))
        for bn, fc in zip(self.bns, self.fcs):
            z = bn(self.dp(F.relu(fc(z))))
        return z

class CBM(nn.Module):
    """
    Concept Bottleneck Model (CBM).
    Supports "soft", "hard", "autoregressive", "embedding".
    """
    def __init__(
        self,
        num_covariates,
        num_concepts,
        num_classes,
        concept_learning="soft",
        training_mode="joint",
        encoder_arch="FCNN",
        head_arch="linear",
        num_monte_carlo=5,
        straight_through=True,
        embedding_size=64
    ):
        super(CBM, self).__init__()

        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.encoder_arch = encoder_arch
        self.head_arch = head_arch
        self.training_mode = training_mode
        self.concept_learning = concept_learning
        self.num_monte_carlo = num_monte_carlo
        self.straight_through = straight_through
        self.embedding_size = embedding_size

        # Build encoder
        if self.encoder_arch == "FCNN":
            n_features = 256
            self.encoder = FCNNEncoder(
                num_inputs=num_covariates, num_hidden=n_features, num_deep=2
            )
        else:
            raise NotImplementedError("Only FCNN architecture is implemented.")

        # Build concept heads based on concept_learning
        if self.concept_learning == "embedding":
            # Concept Embedding Model (CEM)
            self.CEM_embedding = self.embedding_size
            self.positive_embeddings = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(n_features, self.CEM_embedding, bias=True),
                        nn.LeakyReLU(),
                    )
                    for _ in range(self.num_concepts)
                ]
            )
            self.negative_embeddings = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(n_features, self.CEM_embedding, bias=True),
                        nn.LeakyReLU(),
                    )
                    for _ in range(self.num_concepts)
                ]
            )
            self.scoring_function = nn.Sequential(
                nn.Linear(self.CEM_embedding * 2, 1, bias=True), nn.Sigmoid()
            )
            self.concept_dim = self.CEM_embedding * self.num_concepts

        elif self.concept_learning == "autoregressive":
            self.concept_predictor = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(256 + i, 50, bias=True),  # 256 is hidden_dim
                        nn.LeakyReLU(),
                        nn.Linear(50, 1, bias=True),
                    )
                    for i in range(self.num_concepts)
                ]
            )
            self.concept_dim = self.num_concepts

        else:
            # "soft" and "hard"
            self.concept_predictor = nn.Linear(256, self.num_concepts, bias=True)
            self.concept_dim = self.num_concepts

        self.act_c = nn.Sigmoid()  # Activation for concept probabilities

        # Build label head
        if self.num_classes == 2:
            self.pred_dim = 1
        else:
            self.pred_dim = self.num_classes

        if self.head_arch == "linear":
            self.head = nn.Sequential(nn.Linear(self.concept_dim, self.pred_dim))
        else:
            # Optional deeper head
            fc1_y = nn.Linear(self.concept_dim, 256)
            fc2_y = nn.Linear(256, self.pred_dim)
            self.head = nn.Sequential(fc1_y, nn.ReLU(), fc2_y)

    def forward(
        self,
        x,
        epoch=0,
        c_true=None,
        validation=False,
        concepts_train_ar=False,
    ):
        """
        Forward pass of CBM.
        Returns:
        c_prob: Predicted concept probabilities
        y_pred_logits: Predicted logits for target classes
        c: Processed concepts (binary or embeddings)
        """
        # Encode features
        intermediate = self.encoder(x)  # (B, hidden_dim=256)

        # Predict concepts
        if self.concept_learning in ("soft", "hard"):
            c_logit = self.concept_predictor(intermediate)  # (B, k)
            c_prob = self.act_c(c_logit)  # (B, k)

            if self.concept_learning == "hard":
                # Hard predictions using Straight-Through Estimator
                if self.training_mode == "joint" and not validation:
                    c = torch.bernoulli(c_prob)
                    if self.straight_through:
                        c = c_prob + (c - c_prob).detach()
                    else:
                        c = c_prob
                else:
                    # During validation or other modes, use soft probabilities
                    c = c_prob
            else:
                # Soft predictions
                c = c_prob

        elif self.concept_learning == "autoregressive":
            if validation:
                c_prob_list, c_hard_list = [], []
                for i, predictor in enumerate(self.concept_predictor):
                    if len(c_hard_list) == 0:
                        # Initial concept input
                        concept_input = intermediate.unsqueeze(-1).expand(-1, -1, self.num_monte_carlo)  # (B, hidden_dim, MCMC)
                    else:
                        # Concatenate with previously predicted concepts
                        cat_concepts = torch.cat([cc for cc in c_hard_list], dim=1)  # (B, i, MCMC)
                        intermediate_exp = intermediate.unsqueeze(-1).expand(-1, -1, self.num_monte_carlo)  # (B, hidden_dim, MCMC)
                        concept_input = torch.cat([intermediate_exp, cat_concepts], dim=1)  # (B, hidden_dim + i, MCMC)

                    # Reshape for predictor
                    concept_input_flat = concept_input.permute(0, 2, 1).contiguous().view(-1, concept_input.size(1))  # (B*MCMC, hidden_dim + i)

                    # Predict concept i
                    logit_i = predictor(concept_input_flat)  # (B*MCMC, 1)
                    prob_i = self.act_c(logit_i)  # (B*MCMC, 1)

                    # Reshape back
                    prob_i = prob_i.view(-1, 1, self.num_monte_carlo)  # (B, 1, MCMC)

                    # Hard sampling
                    hard_i = torch.bernoulli(prob_i)  # (B, 1, MCMC)

                    if self.straight_through:
                        hard_i = prob_i + (hard_i - prob_i).detach()

                    c_prob_list.append(prob_i)
                    c_hard_list.append(hard_i)

                # Concatenate all concept probabilities and hard samples
                c_prob = torch.cat(c_prob_list, dim=1)  # (B, k, MCMC)
                c = torch.cat(c_hard_list, dim=1)  # (B, k, MCMC)
            else:
                # During training, use probabilities directly
                c_prob = self.act_c(torch.randn(x.size(0), self.num_concepts, device=x.device))
                c = c_prob  # (B, k)

        elif self.concept_learning == "embedding":
            # Concept Embedding Model (CEM)
            c_p = [pe(intermediate) for pe in self.positive_embeddings]  # List of (B, embedding_size)
            c_n = [ne(intermediate) for ne in self.negative_embeddings]  # List of (B, embedding_size)
            c_prob_list, z_list = [], []
            for i in range(self.num_concepts):
                prob_i = self.scoring_function(torch.cat((c_p[i], c_n[i]), dim=1))  # (B,1)
                z_i = prob_i * c_p[i] + (1 - prob_i) * c_n[i]  # (B, embedding_size)
                c_prob_list.append(prob_i)  # List of (B,1)
                z_list.append(z_i)  # List of (B, embedding_size)
            c_prob = torch.cat(c_prob_list, dim=1)  # (B, k)
            c = torch.cat(z_list, dim=1)  # (B, k*embedding_size)

        else:
            raise NotImplementedError(f"Unknown concept_learning={self.concept_learning}.")

        # Prepare input for label prediction
        if self.concept_learning == "hard" and not validation:
            # For hard concepts, average over MCMC dimension if present
            if c_prob.dim() == 3:
                c_input = c_prob.mean(dim=-1)  # (B, k)
            else:
                c_input = c_prob  # (B, k)
        elif self.concept_learning == "autoregressive" and validation:
            if c_prob.dim() == 3:
                c_input = c_prob.mean(dim=-1)  # (B, k)
            else:
                c_input = c_prob  # (B, k)
        else:
            if c_prob.dim() == 3:
                c_input = c_prob.mean(dim=-1)  # (B, k)
            else:
                if self.concept_learning in ("hard", "soft"):
                    c_input = c_prob  # (B, k)
                else:
                    c_input = c  # (B, k*embedding_size)

        # Predict target
        y_pred_logits = self.head(c_input)  # (B, J)

        return c_prob, y_pred_logits, c

# =========================================================
# LOSS FUNCTION
# =========================================================

class CBLoss(nn.Module):
    """
    Combined loss for CBMs:
    - Concept Loss: Binary Cross-Entropy for each concept.
    - Target Loss: Binary Cross-Entropy or Cross-Entropy for target prediction.
    """
    def __init__(self, num_classes=2, reduction="mean", alpha=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, c_pred_probs, c_true, y_logits, y_true):
        """
        Args:
            c_pred_probs: (B, k) or (B, k, MCMC)
            c_true: (B, k)
            y_logits: (B, num_classes) or (B,1)
            y_true: (B,) with 0-based class labels
        Returns:
            target_loss, concept_loss, total_loss
        """
        # Concept Loss
        if c_pred_probs.dim() == 3:
            c_pred_probs_2d = c_pred_probs.mean(dim=-1)  # (B, k)
        else:
            c_pred_probs_2d = c_pred_probs  # (B, k)

        concept_loss = 0.0
        for concept_idx in range(c_true.shape[1]):
            bc = F.binary_cross_entropy(
                c_pred_probs_2d[:, concept_idx],
                c_true[:, concept_idx].float(),
                reduction=self.reduction
            )
            concept_loss += bc
        concept_loss = self.alpha * concept_loss

        # Target Loss
        if self.num_classes == 2:
            # Binary classification
            if y_logits.size(1) == 1:
                y_probs = torch.sigmoid(y_logits.squeeze(1))  # (B,)
            else:
                y_probs = torch.sigmoid(y_logits[:,1])  # (B,)
            target_loss = F.binary_cross_entropy(
                y_probs, y_true.float(), reduction=self.reduction
            )
        else:
            # Multi-class classification
            target_loss = F.cross_entropy(y_logits, y_true.long(), reduction=self.reduction)

        # Total Loss
        total_loss = target_loss + concept_loss
        return target_loss, concept_loss, total_loss

# =========================================================
# DATASET CLASS
# =========================================================

class CBMDataset(Dataset):
    """
    Dataset class for CBM training.
    """
    def __init__(self, X, c, y):
        super().__init__()
        self.X = torch.FloatTensor(X)
        self.c = torch.FloatTensor(c)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "features": self.X[idx],
            "concepts": self.c[idx],
            "labels": self.y[idx],
        }

# =========================================================
# TRAINING FUNCTION
# =========================================================

def train_cbm(
    X_train,
    c_train,
    y_train,
    X_val,
    c_val,
    y_val,
    num_classes=2,
    concept_learning="soft",
    training_mode="joint",
    epochs=10,
    device="cpu",
    alpha=1.0,
    batch_size=64,
    learning_rate=1e-3,
    num_monte_carlo=5,
    straight_through=True,
    embedding_size=64,
):
    """
    Train the CBM based on the specified configuration.
    Args:
        X_train, c_train, y_train: Training data.
        X_val, c_val, y_val: Validation data.
        num_classes: Number of target classes.
        concept_learning: Type of CBM ("soft", "hard", "autoregressive", "embedding").
        training_mode: Training mode ("joint", "sequential", "independent").
        epochs: Number of training epochs.
        device: Device to train on ("cpu" or "cuda").
        alpha: Weighting factor for concept loss.
        batch_size: Number of samples per batch.
        learning_rate: Learning rate for optimizer.
        num_monte_carlo: Number of Monte Carlo samples (for autoregressive).
        straight_through: Whether to use Straight-Through Estimator.
        embedding_size: Size of embeddings (for "embedding" CBM).
    Returns:
        model: Trained CBM model.
        warning_occurred: Boolean indicating if any gradient warnings occurred.
    """
    try:
        # Initialize model with hyperparameters
        model = CBM(
            num_covariates=X_train.shape[1],
            num_concepts=c_train.shape[1],
            num_classes=num_classes,
            concept_learning=concept_learning,
            training_mode=training_mode,
            num_monte_carlo=num_monte_carlo,
            straight_through=straight_through,
            embedding_size=embedding_size
        ).to(device)

        # Ensure all parameters require gradients initially
        for name, param in model.named_parameters():
            param.requires_grad = True

        criterion = CBLoss(num_classes=num_classes, alpha=alpha)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight_decay for regularization

        train_loader = DataLoader(
            CBMDataset(X_train, c_train, y_train),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            CBMDataset(X_val, c_val, y_val),
            batch_size=batch_size,
            shuffle=False
        )

        # Initialize lists to store loss values for visualization
        train_total_losses = []
        val_total_losses = []

        # Initialize a flag to track warnings
        warning_occurred = False  # Flag to track warnings

        if training_mode == "independent":
            # Phase 1: Train Concept Predictors Only
            print("Phase 1: Training Concept Predictors Only")
            for ep in range(epochs):
                model.train()
                for batch in train_loader:
                    x_b = batch["features"].to(device)
                    c_b = batch["concepts"].to(device)

                    c_prob, _, _ = model(x_b, epoch=ep)
                    if c_prob.dim() == 3:
                        c_prob_2d = c_prob.mean(dim=-1)
                    else:
                        c_prob_2d = c_prob

                    # Concept Loss Only
                    concept_loss = 0.0
                    for concept_idx in range(c_b.shape[1]):
                        bc = F.binary_cross_entropy(
                            c_prob_2d[:, concept_idx],
                            c_b[:, concept_idx].float(),
                            reduction='mean'
                        )
                        concept_loss += bc
                    concept_loss *= alpha

                    # Check if concept_loss requires gradients
                    if not concept_loss.requires_grad:
                        print("Warning: concept_loss does not require gradients.")
                        warning_occurred = True
                        continue

                    optimizer.zero_grad()
                    concept_loss.backward()
                    optimizer.step()

                # Evaluate on validation set
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        x_b = batch["features"].to(device)
                        c_b = batch["concepts"].to(device)

                        c_prob, _, _ = model(x_b, epoch=ep, validation=True)
                        if c_prob.dim() == 3:
                            c_prob_2d = c_prob.mean(dim=-1)
                        else:
                            c_prob_2d = c_prob

                        # Concept Loss Only
                        concept_loss = 0.0
                        for concept_idx in range(c_b.shape[1]):
                            bc = F.binary_cross_entropy(
                                c_prob_2d[:, concept_idx],
                                c_b[:, concept_idx].float(),
                                reduction='mean'
                            )
                            concept_loss += bc
                        concept_loss *= alpha
                        val_losses.append(concept_loss.item())
                val_loss_avg = np.mean(val_losses)
                train_total_losses.append(concept_loss.item())
                val_total_losses.append(val_loss_avg)
                print(f"Phase 1 - Epoch {ep+1}/{epochs}, val_concept_loss={val_loss_avg:.4f}")

            # Phase 2: Train Label Predictor Only
            print("Phase 2: Training Label Predictor Only")
            # Freeze Concept Predictors
            freeze_module(model.encoder)
            if concept_learning in ("soft", "hard"):
                freeze_module(model.concept_predictor)
            elif concept_learning == "embedding":
                for pe, ne in zip(model.positive_embeddings, model.negative_embeddings):
                    freeze_module(pe)
                    freeze_module(ne)
                freeze_module(model.scoring_function)
            elif concept_learning == "autoregressive":
                for predictor in model.concept_predictor:
                    freeze_module(predictor)
            else:
                raise NotImplementedError(f"Unknown concept_learning={concept_learning}.")

            # Modify optimizer to only update label head parameters
            optimizer = torch.optim.Adam(model.head.parameters(), lr=learning_rate, weight_decay=1e-5)

            for ep in range(epochs):
                model.train()
                for batch in train_loader:
                    x_b = batch["features"].to(device)
                    c_b = batch["concepts"].to(device)
                    y_b = batch["labels"].to(device)

                    _, y_logits, _ = model(x_b, epoch=ep)
                    if num_classes == 2:
                        if y_logits.size(1) == 1:
                            y_probs = torch.sigmoid(y_logits.squeeze(1))
                        else:
                            y_probs = torch.sigmoid(y_logits[:,1])
                        target_loss = F.binary_cross_entropy(
                            y_probs, y_b.float(), reduction='mean'
                        )
                    else:
                        target_loss = F.cross_entropy(y_logits, y_b.long(), reduction='mean')

                    # Check if target_loss requires gradients
                    if not target_loss.requires_grad:
                        print("Warning: target_loss does not require gradients.")
                        warning_occurred = True
                        continue

                    optimizer.zero_grad()
                    target_loss.backward()
                    optimizer.step()

                # Evaluate on validation set
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        x_b = batch["features"].to(device)
                        c_b = batch["concepts"].to(device)
                        y_b = batch["labels"].to(device)

                        _, y_logits, _ = model(x_b, epoch=ep, validation=True)
                        if num_classes == 2:
                            if y_logits.size(1) == 1:
                                y_probs = torch.sigmoid(y_logits.squeeze(1))
                            else:
                                y_probs = torch.sigmoid(y_logits[:,1])
                            target_loss = F.binary_cross_entropy(
                                y_probs, y_b.float(), reduction='mean'
                            )
                        else:
                            target_loss = F.cross_entropy(y_logits, y_b.long(), reduction='mean')
                        val_losses.append(target_loss.item())
                val_loss_avg = np.mean(val_losses)
                train_total_losses.append(target_loss.item())
                val_total_losses.append(val_loss_avg)
                print(f"Phase 2 - Epoch {ep+1}/{epochs}, val_label_loss={val_loss_avg:.4f}")

            return model, warning_occurred

        elif training_mode == "joint":
            # Jointly train concept predictors and label predictor
            print("Training Mode: Joint")
            for ep in range(epochs):
                model.train()
                for batch in train_loader:
                    x_b = batch["features"].to(device)
                    c_b = batch["concepts"].to(device)
                    y_b = batch["labels"].to(device)

                    c_prob, y_logits, _ = model(x_b, epoch=ep)
                    target_loss, concept_loss, total_loss = criterion(c_prob, c_b, y_logits, y_b)

                    # Check if total_loss requires gradients
                    if not total_loss.requires_grad:
                        print("Warning: total_loss does not require gradients.")
                        warning_occurred = True
                        continue

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                # Evaluate on validation set
                model.eval()
                val_target_losses = []
                val_concept_losses = []
                val_total_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        x_b = batch["features"].to(device)
                        c_b = batch["concepts"].to(device)
                        y_b = batch["labels"].to(device)

                        c_prob, y_logits, _ = model(x_b, epoch=ep, validation=True)
                        target_loss, concept_loss, total_loss = criterion(c_prob, c_b, y_logits, y_b)
                        val_target_losses.append(target_loss.item())
                        val_concept_losses.append(concept_loss.item())
                        val_total_losses.append(total_loss.item())
                val_target_avg = np.mean(val_target_losses)
                val_concept_avg = np.mean(val_concept_losses)
                val_total_avg = np.mean(val_total_losses)
                train_total_losses.append(total_loss.item())
                val_total_losses.append(val_total_avg)
                print(f"Epoch {ep+1}/{epochs} - val_target_loss={val_target_avg:.4f}, val_concept_loss={val_concept_avg:.4f}, val_total_loss={val_total_avg:.4f}")

            return model, warning_occurred

        elif training_mode == "sequential":
            # Sequentially train concept predictors and then label predictor
            print("Training Mode: Sequential")
            # Phase 1: Train Concept Predictors Only
            print("Phase 1: Training Concept Predictors Only")
            for ep in range(epochs):
                model.train()
                for batch in train_loader:
                    x_b = batch["features"].to(device)
                    c_b = batch["concepts"].to(device)

                    c_prob, _, _ = model(x_b, epoch=ep)
                    if c_prob.dim() == 3:
                        c_prob_2d = c_prob.mean(dim=-1)
                    else:
                        c_prob_2d = c_prob

                    # Concept Loss Only
                    concept_loss = 0.0
                    for concept_idx in range(c_b.shape[1]):
                        bc = F.binary_cross_entropy(
                            c_prob_2d[:, concept_idx],
                            c_b[:, concept_idx].float(),
                            reduction='mean'
                        )
                        concept_loss += bc
                    concept_loss *= alpha

                    # Check if concept_loss requires gradients
                    if not concept_loss.requires_grad:
                        print("Warning: concept_loss does not require gradients.")
                        warning_occurred = True
                        continue

                    optimizer.zero_grad()
                    concept_loss.backward()
                    optimizer.step()

                # Evaluate on validation set
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        x_b = batch["features"].to(device)
                        c_b = batch["concepts"].to(device)

                        c_prob, _, _ = model(x_b, epoch=ep, validation=True)
                        if c_prob.dim() == 3:
                            c_prob_2d = c_prob.mean(dim=-1)
                        else:
                            c_prob_2d = c_prob

                        # Concept Loss Only
                        concept_loss = 0.0
                        for concept_idx in range(c_b.shape[1]):
                            bc = F.binary_cross_entropy(
                                c_prob_2d[:, concept_idx],
                                c_b[:, concept_idx].float(),
                                reduction='mean'
                            )
                            concept_loss += bc
                        concept_loss *= alpha
                        val_losses.append(concept_loss.item())
                val_loss_avg = np.mean(val_losses)
                train_total_losses.append(concept_loss.item())
                val_total_losses.append(val_loss_avg)
                print(f"Phase 1 - Epoch {ep+1}/{epochs}, val_concept_loss={val_loss_avg:.4f}")

            # Phase 2: Train Label Predictor Only
            print("Phase 2: Training Label Predictor Only")
            # Freeze Concept Predictors
            freeze_module(model.encoder)
            if concept_learning in ("soft", "hard"):
                freeze_module(model.concept_predictor)
            elif concept_learning == "embedding":
                for pe, ne in zip(model.positive_embeddings, model.negative_embeddings):
                    freeze_module(pe)
                    freeze_module(ne)
                freeze_module(model.scoring_function)
            elif concept_learning == "autoregressive":
                for predictor in model.concept_predictor:
                    freeze_module(predictor)
            else:
                raise NotImplementedError(f"Unknown concept_learning={concept_learning}.")

            # Modify optimizer to only update label head parameters
            optimizer = torch.optim.Adam(model.head.parameters(), lr=learning_rate, weight_decay=1e-5)

            for ep in range(epochs):
                model.train()
                for batch in train_loader:
                    x_b = batch["features"].to(device)
                    c_b = batch["concepts"].to(device)
                    y_b = batch["labels"].to(device)

                    _, y_logits, _ = model(x_b, epoch=ep)
                    if num_classes == 2:
                        if y_logits.size(1) == 1:
                            y_probs = torch.sigmoid(y_logits.squeeze(1))
                        else:
                            y_probs = torch.sigmoid(y_logits[:,1])
                        target_loss = F.binary_cross_entropy(
                            y_probs, y_b.float(), reduction='mean'
                        )
                    else:
                        target_loss = F.cross_entropy(y_logits, y_b.long(), reduction='mean')

                    # Check if target_loss requires gradients
                    if not target_loss.requires_grad:
                        print("Warning: target_loss does not require gradients.")
                        warning_occurred = True
                        continue

                    optimizer.zero_grad()
                    target_loss.backward()
                    optimizer.step()

                # Evaluate on validation set
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        x_b = batch["features"].to(device)
                        c_b = batch["concepts"].to(device)
                        y_b = batch["labels"].to(device)

                        _, y_logits, _ = model(x_b, epoch=ep, validation=True)
                        if num_classes == 2:
                            if y_logits.size(1) == 1:
                                y_probs = torch.sigmoid(y_logits.squeeze(1))
                            else:
                                y_probs = torch.sigmoid(y_logits[:,1])
                            target_loss = F.binary_cross_entropy(
                                y_probs, y_b.float(), reduction='mean'
                            )
                        else:
                            target_loss = F.cross_entropy(y_logits, y_b.long(), reduction='mean')
                        val_losses.append(target_loss.item())
                val_loss_avg = np.mean(val_losses)
                train_total_losses.append(target_loss.item())
                val_total_losses.append(val_loss_avg)
                print(f"Phase 2 - Epoch {ep+1}/{epochs}, val_label_loss={val_loss_avg:.4f}")

            return model, warning_occurred

        else:
            raise NotImplementedError(f"Unknown training_mode={training_mode}.")

    except Exception as e:
        print(f"An exception occurred during training: {e}")
        return None, False

# =========================================================
# LEAKAGE ESTIMATION FUNCTIONS
# =========================================================

def get_cbm_concepts(model, X, c, y, device="cpu"):
    """
    Get predicted concepts from the CBM.
    Args:
        model: Trained CBM model.
        X: Features.
        c: Ground-truth concepts (unused here).
        y: Labels (unused here).
        device: Device to run inference on.
    Returns:
        c_hat_all: (n, k) predicted concept probabilities.
    """
    model.eval()
    ds = CBMDataset(X, c, y)
    loader = DataLoader(ds, batch_size=128, shuffle=False)

    all_c_hat = []
    with torch.no_grad():
        for batch in loader:
            x_b = batch["features"].to(device)
            c_prob, _, _ = model(x_b, validation=True)
            # Average over MCMC dimension if present
            if c_prob.dim() == 3:
                c_prob = c_prob.mean(dim=-1)
            all_c_hat.append(c_prob.cpu().numpy())
    return np.concatenate(all_c_hat, axis=0)

def train_xgb_and_get_nll(X_train, y_train, X_val, y_val, X_test, y_test, num_classes, seed, learning_rate=1e-3, num_rounds=100):
    """
    Train XGBoost on given data and compute Negative Log-Likelihood (NLL) on test set.
    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        X_test, y_test: Test data.
        num_classes: Number of classes.
        seed: Random seed for reproducibility.
        learning_rate: Learning rate for XGBoost.
        num_rounds: Number of boosting rounds.
    Returns:
        nll_test: Negative log-likelihood on test set.
    """
    clf = xgb.XGBClassifier(
        eval_metric="mlogloss",
        num_class=num_classes,
        use_label_encoder=False,
        random_state=seed,
        learning_rate=learning_rate,
        n_estimators=num_rounds
    )

    # Check if all classes are present in y_train
    unique_classes = np.unique(y_train)
    expected_classes = np.arange(num_classes)
    missing_classes = expected_classes[~np.isin(expected_classes, unique_classes)]
    if len(missing_classes) > 0:
        print(f"Warning: Training data is missing classes {missing_classes}. Adding dummy samples to include them.")
        # Add dummy samples for missing classes
        for cls in missing_classes:
            X_dummy = np.random.normal(0, 1, size=(10, X_train.shape[1]))
            y_dummy = np.full(10, cls)
            X_train = np.vstack([X_train, X_dummy])
            y_train = np.hstack([y_train, y_dummy])

    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    p_test = clf.predict_proba(X_test)
    # log_loss expects y_test in [0..num_classes-1]
    nll_test = log_loss(y_test, p_test, labels=list(range(num_classes)))
    return nll_test

def estimate_leakage_with_xgb(
    num_sim=3,
    n=1000,
    d=500,
    k=10,
    J=5,
    b=200,
    l=0,
    concept_learning="soft",
    training_mode="joint",
    device="cpu",
    seed=42,
    epochs=5,
    alpha=1.0,
    batch_size=64,
    learning_rate=1e-3,
    num_monte_carlo=5,
    straight_through=True,
    embedding_size=64,
    config_id=1,
    total_configs=10,
):
    """
    Simulate leakage for a given CBM configuration.
    Args:
        num_sim: Number of simulation runs.
        n, d, k, J, b, l: Data generation parameters.
        concept_learning: CBM type.
        training_mode: Training mode.
        device: Device to run training on.
        seed: Random seed.
        epochs: Number of training epochs.
        alpha: Weighting factor for concept loss.
        batch_size: Number of samples per batch.
        learning_rate: Learning rate for optimizer.
        num_monte_carlo: Number of Monte Carlo samples (for autoregressive).
        straight_through: Whether to use Straight-Through Estimator.
        embedding_size: Size of embeddings (for "embedding" CBM).
        config_id: Current configuration ID (for logging).
        total_configs: Total number of configurations (for logging).
    Returns:
        all_leakage: List of leakage values across simulations.
        all_warnings: List indicating if warnings occurred in each simulation.
    """
    rng = np.random.RandomState(seed)
    all_leakage = []
    all_warnings = []

    for sim_i in range(num_sim):
        sim_seed = rng.randint(0, 10_000_000)

        # 1) Generate data
        X, c_true, _, y_raw = generate_synthetic_data_leakage(
            n=n,
            d=d,
            k=k,
            J=J,
            b=b,
            l=l,
            seed=sim_seed
        )
        # Convert y from 1-based to 0-based (already adjusted in data generation)
        y = y_raw.astype(int)  # Should be in [0, J-1]

        # 2) Split into train/val/test
        idxs = np.arange(n)
        rng.shuffle(idxs)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        idx_train = idxs[:n_train]
        idx_val = idxs[n_train:n_train + n_val]
        idx_test = idxs[n_train + n_val:]

        X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
        c_train, c_val, c_test = c_true[idx_train], c_true[idx_val], c_true[idx_test]
        y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

        # 3) Train CBM
        cbm_model, warning_occurred = train_cbm(
            X_train, c_train, y_train,
            X_val, c_val, y_val,
            num_classes=J if J > 2 else 2,
            concept_learning=concept_learning,
            training_mode=training_mode,
            epochs=epochs,
            device=device,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_monte_carlo=num_monte_carlo,
            straight_through=straight_through,
            embedding_size=embedding_size
        )

        if cbm_model is None:
            print(f"[Config {config_id}/{total_configs}] [Sim {sim_i+1}/{num_sim}] Error: CBM model is None.")
            all_leakage.append(np.nan)
            all_warnings.append(True)
            continue

        # 4) Get c_hat for entire dataset
        c_hat_all = get_cbm_concepts(cbm_model, X, c_true, y, device=device)
        c_hat_train = c_hat_all[idx_train]
        c_hat_val = c_hat_all[idx_val]
        c_hat_test = c_hat_all[idx_test]

        # 5) Train XGBoost on true concepts (c) => NLL_gb
        nll_gb = train_xgb_and_get_nll(
            X_train=c_train,
            y_train=y_train,
            X_val=c_val,
            y_val=y_val,
            X_test=c_test,
            y_test=y_test,
            num_classes=J,
            seed=sim_seed,
            learning_rate=learning_rate,
            num_rounds=100
        )

        # 6) Train XGBoost on [c_hat, c] => NLL_ga
        X_train_ga = np.hstack([c_hat_train, c_train])
        X_val_ga = np.hstack([c_hat_val, c_val])
        X_test_ga = np.hstack([c_hat_test, c_test])

        nll_ga = train_xgb_and_get_nll(
            X_train=X_train_ga,
            y_train=y_train,
            X_val=X_val_ga,
            y_val=y_val,
            X_test=X_test_ga,
            y_test=y_test,
            num_classes=J,
            seed=sim_seed,
            learning_rate=learning_rate,
            num_rounds=100
        )

        leakage = nll_gb - nll_ga
        all_leakage.append(leakage)
        all_warnings.append(warning_occurred)
        print(f"[Config {config_id}/{total_configs}] [Sim {sim_i+1}/{num_sim}] NLL_gb={nll_gb:.4f}, NLL_ga={nll_ga:.4f}, leakage={leakage:.4f}, Warning={warning_occurred}")

    return all_leakage, all_warnings

# =========================================================
# EXPERIMENT FUNCTION
# =========================================================

def run_experiments(
    configurations,
    concept_range,
    alpha_range=None,  # List of alpha values for "joint" training mode
    training_modes=None,  # List of training modes
    num_sim=3,
    n=1000,
    d=500,
    J=5,
    b=200,
    l=0,
    noise_level=None,  # Not directly used; placeholder for future enhancements
    central_seed=42,
    epochs=5,
    alpha_default=1.0,
    device="cpu",
    batch_size=64,
    learning_rate=1e-3,
    num_monte_carlo=5,
    straight_through=True,
    embedding_size=64,
):
    """
    Run experiments based on specified configurations and collect leakage estimates.
    Args:
        configurations: List of CBM types (e.g., ["soft", "hard", "autoregressive", "embedding"]).
        concept_range: List of k values (number of concepts) to experiment with.
        alpha_range: List of alpha values to test for "joint" training mode.
        training_modes: List of training modes (e.g., ["joint", "sequential", "independent"]).
        num_sim: Number of simulation runs per configuration.
        n, d, J, b, l: Data generation parameters.
        noise_level: Not directly used; placeholder for future enhancements.
        central_seed: Central seed for reproducibility.
        epochs: Number of training epochs for CBM.
        alpha_default: Default weighting factor for concept loss.
        device: "cpu" or "cuda".
        batch_size: Number of samples per batch.
        learning_rate: Learning rate for optimizer.
        num_monte_carlo: Number of Monte Carlo samples (for autoregressive).
        straight_through: Whether to use Straight-Through Estimator.
        embedding_size: Size of embeddings (for "embedding" CBM).
    Returns:
        leakage_df: pandas DataFrame containing leakage results for all runs.
    """
    # Initialize results list
    results = []

    # Define default training modes if not provided
    if training_modes is None:
        training_modes = ["joint", "sequential", "independent"]

    # Calculate total number of configurations correctly
    total_configs = 0
    for cbm_type in configurations:
        applicable_training_modes = []
        if cbm_type in ["soft", "hard"]:
            applicable_training_modes = training_modes  # All modes
        else:
            applicable_training_modes = ["joint"]  # Only joint mode

        if cbm_type == "soft" and alpha_range is not None:
            current_alpha_values = alpha_range
        else:
            current_alpha_values = [alpha_default]

        for training_mode in applicable_training_modes:
            for alpha_val in current_alpha_values:
                for k in concept_range:
                    total_configs += 1

    config_counter = 1  # To track the current configuration number

    # Iterate over each CBM type
    for cbm_type in configurations:
        # Determine applicable training modes
        if cbm_type in ["soft", "hard"]:
            applicable_training_modes = training_modes  # All modes
        else:
            applicable_training_modes = ["joint"]  # Only joint mode

        # Determine alpha values
        if cbm_type == "soft" and alpha_range is not None:
            current_alpha_values = alpha_range
        else:
            current_alpha_values = [alpha_default]

        # Iterate over each training mode
        for training_mode in applicable_training_modes:
            # Iterate over each alpha value
            for alpha_val in current_alpha_values:
                # Iterate over each concept value
                for k in concept_range:
                    # Check if k exceeds projection limits
                    if k > b or k > (d - b - l):
                        print(f"\n=== Skipping Configuration {config_counter}/{total_configs}: CBM Type='{cbm_type}', Training Mode='{training_mode}', Alpha={'N/A' if not (cbm_type == 'soft' and training_mode == 'joint') else alpha_val}, k={k} ===")
                        print(f"Reason: k={k} exceeds projection limits (k <= {min(b, d - b - l)}).")
                        # Log the skipped configuration
                        results.append({
                            'CBM_Type': cbm_type,
                            'Training_Mode': training_mode,
                            'Alpha': alpha_val if (cbm_type == "soft" and training_mode == "joint") else np.nan,
                            'k': k,
                            'Run': np.nan,
                            'Leakage': np.nan,
                            'Warning': True,  # Mark as warning since it's skipped
                            'num_sim': num_sim,
                            'n': n,
                            'd': d
                        })
                        config_counter += 1
                        continue  # Skip to next configuration

                    # Determine display value for alpha
                    if cbm_type == "soft" and training_mode == "joint":
                        alpha_display = alpha_val
                    else:
                        alpha_display = "N/A"

                    print(f"\n=== Running Configuration {config_counter}/{total_configs}: CBM Type='{cbm_type}', Training Mode='{training_mode}', Alpha={alpha_display}, k={k} ===")

                    # Run simulations
                    leakages, warnings = estimate_leakage_with_xgb(
                        num_sim=num_sim,
                        n=n,
                        d=d,
                        k=k,
                        J=J,
                        b=b,
                        l=l,
                        concept_learning=cbm_type,
                        training_mode=training_mode,
                        device=device,
                        seed=central_seed,  # Central seed
                        epochs=epochs,
                        alpha=alpha_val,  # Assign current alpha value
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        num_monte_carlo=num_monte_carlo,
                        straight_through=straight_through,
                        embedding_size=embedding_size,
                        config_id=config_counter,
                        total_configs=total_configs
                    )

                    # Collect results
                    for run_idx, (leakage, warning) in enumerate(zip(leakages, warnings), start=1):
                        results.append({
                            'CBM_Type': cbm_type,
                            'Training_Mode': training_mode,
                            'Alpha': alpha_val if (cbm_type == "soft" and training_mode == "joint") else np.nan,
                            'k': k,
                            'Run': run_idx,
                            'Leakage': leakage,
                            'Warning': warning,
                            'num_sim': num_sim,
                            'n': n,
                            'd': d
                        })

                    config_counter += 1

    # Convert results to DataFrame
    leakage_df = pd.DataFrame(results)
    return leakage_df

# =========================================================
# MAIN FUNCTION
# =========================================================

def main():
    # =====================================================
    # HYPERPARAMETER CONFIGURATION
    # =====================================================
    # Number of simulation runs per configuration
    num_sim = 15  # Specify desired number of simulation runs here

    # Data generation parameters
    n = 2000  # Number of observations
    d = 700  # Feature dimensionality
    J = 5  # Number of target classes
    b = 250  # Number of features used in ground truth concepts
    l = 0  # Number of features excluded from leakage

    # CBM training parameters
    epochs = 5  # Number of training epochs for CBM
    alpha_default = 200  # Default weighting factor for concept loss
    batch_size = 64  # Number of samples per batch
    learning_rate = 1e-3  # Learning rate for optimizer
    num_monte_carlo = 10  # Number of Monte Carlo samples (for autoregressive)
    straight_through = True  # Whether to use Straight-Through Estimator
    embedding_size = 64  # Size of embeddings (for "embedding" CBM)

    # Device configuration
    device = "cpu"  # Set to "cuda" if GPU is available and desired

    # Random seed for reproducibility
    central_seed = 42

    # Define experimental configurations (CBM Types)
    configurations = [
        'soft',  # Joint Soft CBM with varying alpha
        'hard',  # Joint Hard CBM with varying alpha
        'autoregressive',  # Autoregressive CBM
        'embedding'  # Concept Embedding Model
    ]

    # Define training modes
    training_modes = [
        'joint',
        'sequential',
        'independent'
    ]

    # Define range of concept values (k) to experiment with
    concept_range = [10, 20, 50, 100, 200]  # Desired k values

    # Define range of alpha values for "joint" training mode
    alpha_range = [1, 10, 20, 50, 100, 200]

    # =====================================================
    # RUN EXPERIMENTS
    # =====================================================
    leakage_results = run_experiments(
        configurations=configurations,
        concept_range=concept_range,
        alpha_range=alpha_range,  # Passing the range of alpha values
        training_modes=training_modes,
        num_sim=num_sim,
        n=n,
        d=d,
        J=J,
        b=b,
        l=l,
        noise_level=None,  # Placeholder if needed
        central_seed=central_seed,
        epochs=epochs,
        alpha_default=alpha_default,
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_monte_carlo=num_monte_carlo,
        straight_through=straight_through,
        embedding_size=embedding_size
    )

    # =====================================================
    # SAVE RESULTS TO CSV WITH TIMESTAMP
    # =====================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "simulation_results/res_leak_concept_methods/"
    os.makedirs(results_dir, exist_ok=True)  # Create directory if it doesn't exist

    results_path = f"{results_dir}res_leak_concept_methods_{timestamp}.csv"
    leakage_results.to_csv(results_path, index=False)
    print(f"\nFinal results saved to: {results_path}")


    # Compute mean leakage per configuration
    # =================== Modification Below ====================
    mean_leakage = leakage_results.groupby(
        ['CBM_Type', 'Training_Mode', 'Alpha', 'k', 'num_sim', 'n', 'd'],
        dropna=False  # Include groups with NaN Alpha
    )['Leakage'].mean().reset_index()
    # ================================================================

    # Save mean leakage to CSV
    mean_leakage_path = f"{results_dir}res_leak_concept_methods_mean_{timestamp}.csv"
    mean_leakage.to_csv(mean_leakage_path, index=False)
    print(f"Mean leakage results saved to: {mean_leakage_path}")

    # # =====================================================
    # # DISPLAY RESULTS
    # # =====================================================
    # print("\n=== Leakage Results ===")
    # print(leakage_results)

    # print("\n=== Mean Leakage per Configuration ===")
    # print(mean_leakage)

    # # =====================================================
    # # IDENTIFY CONFIGURATIONS WITH WARNINGS
    # # =====================================================
    # problematic_configs = leakage_results[leakage_results['Warning'] == True]
    # print("\n=== Configurations with Gradient Warnings or Skipped ===")
    # if not problematic_configs.empty:
    #     print(problematic_configs)
    # else:
    #     print("No configurations triggered gradient warnings or were skipped.")

    # # =====================================================
    # # VISUALIZE RESULTS
    # # =====================================================
    # # Note: All plots can now be reproduced from the saved CSV files.

    # # 1. Leakage vs. Alpha for Joint Soft CBM across different k values
    # joint_soft_configs = mean_leakage[
    #     (mean_leakage['CBM_Type'] == 'soft') &
    #     (mean_leakage['Training_Mode'] == 'joint')
    # ]

    # if not joint_soft_configs.empty:
    #     plt.figure(figsize=(12, 6))
    #     sns.lineplot(data=joint_soft_configs, x='Alpha', y='Leakage', hue='k', marker='o')
    #     plt.xscale('log')  # Since alpha is on a log scale
    #     plt.title('Leakage vs Alpha for Joint Soft CBM across Different k Values')
    #     plt.xlabel('Alpha (log scale)')
    #     plt.ylabel('Leakage (NLL_gb - NLL_ga)')
    #     plt.legend(title='k (Number of Concepts)')
    #     plt.tight_layout()
    #     plt.show()
    # else:
    #     print("\nNo data available for Joint Soft CBM configurations to plot 'Leakage vs Alpha'.")

    # # 2. Leakage across All Configurations for different k values
    # plt.figure(figsize=(16, 8))
    # sns.boxplot(x='CBM_Type', y='Leakage', hue='Training_Mode', data=leakage_results)
    # plt.title('Leakage across CBM Types and Training Modes')
    # plt.xlabel('CBM Type')
    # plt.ylabel('Leakage (NLL_gb - NLL_ga)')
    # plt.legend(title='Training Mode')
    # plt.tight_layout()
    # plt.show()

    # # 3. Leakage across All Configurations and k values
    # plt.figure(figsize=(16, 8))
    # sns.scatterplot(data=mean_leakage, x='k', y='Leakage', hue='CBM_Type', style='Training_Mode', s=100)
    # plt.title('Leakage across All Configurations and k Values')
    # plt.xlabel('Number of Concepts (k)')
    # plt.ylabel('Leakage (NLL_gb - NLL_ga)')
    # plt.legend(title='CBM Type / Training Mode')
    # plt.tight_layout()
    # plt.show()

    # # 4. Heatmap of Leakage for Joint Soft CBM
    # if not joint_soft_configs.empty:
    #     pivot_table = joint_soft_configs.pivot(index='k', columns='Alpha', values='Leakage')
    #     plt.figure(figsize=(12, 8))
    #     sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="viridis")
    #     plt.xscale('log')  # Since alpha is on a log scale
    #     plt.title('Heatmap of Leakage for Joint Soft CBM across Alpha and k')
    #     plt.xlabel('Alpha (log scale)')
    #     plt.ylabel('Number of Concepts (k)')
    #     plt.tight_layout()
    #     plt.show()
    # else:
    #     print("\nNo data available for Joint Soft CBM configurations to plot 'Leakage vs Alpha' Heatmap.")

    # # =====================================================
    # # FINAL NOTES
    # # =====================================================
    # print("\n=== Final Notes ===")
    # print("All results have been saved with a timestamp in the 'simulation_results/res_leak_concept_methods/' directory.")
    # print("You can reproduce the plots using the saved CSV files independently.")
    # print("Leakage estimates are relatively sensitive to parameters such as 'alpha', 'num_sim', and 'num_monte_carlo'.")
    # print("Further investigation and parameter tuning are recommended to optimize leakage estimates.")
    # print("Consider conducting more ablation studies on parameters like learning rate, batch size, and model architecture.")
    # print("Ensure that the training sets contain all classes to avoid skewed results.")
    # print("Additional engineering efforts may be required to stabilize training and improve leakage estimates.")
    # print("Remember that significant portions of this code were adapted from the SCBM repository, ensuring proper attribution.")

# =========================================================
# RUN MAIN FUNCTION
# =========================================================

if __name__ == "__main__":
    main()

 