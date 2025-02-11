import os
import math
import torch
from torch import nn
from torch.distributions import RelaxedBernoulli
import torch.nn.functional as F
from torchvision import models


def create_model(config):
    """
    Parse the configuration file and return a relevant model
    """
    if config.model.model == "cbm":
        return CBM(config)
    elif config.model.model == "scbm":
        return SCBM(config)
    else:
        print("Could not create model with name ", config.model, "!")
        quit()


# utils func
def freeze_module(m):
    m.eval()
    for param in m.parameters():
        param.requires_grad = False

# utils func
def unfreeze_module(m):
    m.train()
    for param in m.parameters():
        param.requires_grad = True


# CBM NN architecture
class FCNNEncoder(nn.Module):
    """
    Defining the concept encoder for the synthetic dataset.
    """

    def __init__(self, num_inputs: int, num_hidden: int, num_deep: int):
        super(FCNNEncoder, self).__init__()

        self.fc0 = nn.Linear(num_inputs, num_hidden)
        self.bn0 = nn.BatchNorm1d(num_hidden)
        self.fcs = nn.ModuleList(
            [nn.Linear(num_hidden, num_hidden) for _ in range(num_deep)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(num_deep)])
        self.dp = nn.Dropout(0.05)

    def forward(self, x):
        z = self.bn0(self.dp(F.relu(self.fc0(x))))
        for bn, fc in zip(self.bns, self.fcs):
            z = bn(self.dp(F.relu(fc(z))))
        return z


class CBM(nn.Module):
    """
    Model class encompassing all baselines: Hard & Soft Concept Bottleneck Model (CBM),
                                            Concept Embedding Model (CEM), and Autoregressive CBM (AR).

    This class implements the baselines. Depending on the choice of model, only a small part of the full code is used.
    Check the if statements in the forward method to see which part of the code is used for which model.

    Args:
        config (dict): Configuration dictionary containing model and data settings.

    Noteworthy Attributes:
        training_mode (str): The training mode (e.g., "joint", "sequential", "independent").
        concept_learning (str): The concept learning method ("hard", "soft", "embedding", or "autoregressive").
                                This determines the type of method to use
        num_monte_carlo (int): The number of Monte Carlo samples for sampling Gumbel Softmax in AR.
        straight_through (bool): Flag indicating whether to use straight-through gradients.
        curr_temp (float): The current temperature for the Gumbel-Softmax distribution.
    """

    def __init__(self, config):
        super(CBM, self).__init__()

        # Configuration arguments
        config_model = config.model
        self.num_concepts = config.data.num_concepts
        self.num_classes = config.data.num_classes
        self.encoder_arch = config_model.encoder_arch
        self.head_arch = config_model.head_arch
        self.training_mode = config_model.training_mode
        self.concept_learning = config_model.concept_learning
        if self.concept_learning in ("hard", "autoregressive"):
            self.num_monte_carlo = config_model.num_monte_carlo
            self.straight_through = config_model.straight_through
            self.curr_temp = 1.0
            if self.training_mode == "joint":
                self.num_epochs = config_model.j_epochs
            else:
                self.num_epochs = config_model.t_epochs
        elif self.concept_learning == "embedding":
            self.CEM_embedding = config_model.embedding_size

        # Architectures
        # Encoder h(.)
        if self.encoder_arch == "FCNN":
            n_features = 256
            self.encoder = FCNNEncoder(
                num_inputs=config.data.num_covariates, num_hidden=n_features, num_deep=2
            )
        elif self.encoder_arch == "resnet18":
            self.encoder_res = models.resnet18(weights=None)
            self.encoder_res.load_state_dict(
                torch.load(
                    os.path.join(
                        config_model.model_directory, "resnet/resnet18-5c106cde.pth"
                    )
                )
            )
            n_features = self.encoder_res.fc.in_features
            self.encoder_res.fc = Identity()
            self.encoder = nn.Sequential(self.encoder_res)

        elif self.encoder_arch == "simple_CNN":
            n_features = 256
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, 3),
                nn.ReLU(),
                nn.Conv2d(32, 64, 5, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                nn.Flatten(),
                nn.Linear(9216, n_features),
                nn.ReLU(),
            )

        else:
            raise NotImplementedError("ERROR: architecture not supported!")
        if self.concept_learning == "embedding":
            print(
                "Please be aware that our implementation of CEMs is without training on interventions! This is because we would deem this an unfair comparison to our method that is also not trained on interventions. Still, be careful when using this CEM code for derivative works"
            )
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
        else:
            if self.concept_learning == "autoregressive":
                self.concept_predictor = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(n_features + i, 50, bias=True),
                            nn.LeakyReLU(),
                            nn.Linear(50, 1, bias=True),
                        )
                        for i in range(self.num_concepts)
                    ]
                )

            else:
                self.concept_predictor = nn.Linear(
                    n_features, self.num_concepts, bias=True
                )
            self.concept_dim = self.num_concepts

        # Assume binary concepts
        self.act_c = nn.Sigmoid()

        # Link function g(.)
        if self.num_classes == 2:
            self.pred_dim = 1
        elif self.num_classes > 2:
            self.pred_dim = self.num_classes

        if self.head_arch == "linear":
            fc_y = nn.Linear(self.concept_dim, self.pred_dim)
            self.head = nn.Sequential(fc_y)
        else:
            fc1_y = nn.Linear(self.concept_dim, 256)
            fc2_y = nn.Linear(256, self.pred_dim)
            self.head = nn.Sequential(fc1_y, nn.ReLU(), fc2_y)

    def forward(
        self,
        x,
        epoch,
        c_true=None,
        validation=False,
        concepts_train_ar=False,
    ):
        """
        Perform a forward pass through one of the baselines.

        This method performs a forward pass predicting concept probabilities and logits for the target variable.
        It handles different concept learning strategies and training modes, including hard, soft, autoregressive, and embedding-based concepts.

        Args:
            x (torch.Tensor): The input covariates. Shape: (batch_size, input_dims)
            epoch (int): The current epoch number.
            c_true (torch.Tensor, optional): The ground-truth concept values. Required for "independent" training mode. Default is None.
            validation (bool, optional): Flag indicating whether this is a validation pass. Default is False.
            concepts_train_ar (torch.Tensor, optional): Ground-truth concept values for autoregressive training. Default is False.

        Returns:
            tuple: A tuple containing:
                - c_prob (torch.Tensor): Predicted concept probabilities. Shape: (batch_size, num_concepts)
                - y_pred_logits (torch.Tensor): Logits for the target variable. Shape: (batch_size, label_dim)
                - c (torch.Tensor): Predicted hard concept values (if method permits, otherwise the concept representation). Shape: (batch_size, num_concepts, num_monte_carlo) for MCMC sampling or (batch_size, num_concepts) otherwise.
        """

        # Get intermediate representations
        intermediate = self.encoder(x)

        # Get concept predictions
        if self.concept_learning in ("hard", "soft"):
            # CBM
            c_logit = self.concept_predictor(intermediate)
            c_prob = self.act_c(c_logit)

            if self.concept_learning in ("hard"):
                # Hard CBM
                if self.training_mode == "sequential" or validation:
                    # Sample from Bernoulli M times, as we don't need to backprop
                    c_prob_mcmc = c_prob.unsqueeze(-1).expand(
                        -1, -1, self.num_monte_carlo
                    )
                    c = torch.bernoulli(c_prob_mcmc)

                # Relax bernoulli sampling with Gumbel Softmax to allow for backpropagation
                elif self.training_mode == "joint":
                    curr_temp = self.compute_temperature(epoch, device=c_prob.device)
                    dist = RelaxedBernoulli(temperature=curr_temp, probs=c_prob)
                    c_relaxed = dist.rsample([self.num_monte_carlo]).movedim(0, -1)
                    if self.straight_through:
                        # Straight-Through Gumbel Softmax
                        c_hard = (c_relaxed > 0.5) * 1
                        c = c_hard - c_relaxed.detach() + c_relaxed
                    else:
                        # Reparametrization trick.
                        c = c_relaxed

                else:
                    raise NotImplementedError

        elif self.concept_learning == "autoregressive":
            # AR
            if validation:
                c_prob, c_hard = [], []
                for predictor in self.concept_predictor:
                    if c_prob:
                        concept = []
                        for i in range(
                            self.num_monte_carlo
                        ):  # MCMC samples for evaluation and interventions, but not for training
                            concept_input_i = torch.cat(
                                [intermediate, torch.cat(c_hard, dim=1)[..., i]], dim=1
                            )
                            concept.append(self.act_c(predictor(concept_input_i)))
                        concept = torch.cat(concept, dim=-1)
                        c_relaxed = torch.bernoulli(concept)[:, None, :]
                        concept = concept[:, None, :]
                        concept_hard = c_relaxed

                    else:
                        concept_input = intermediate
                        concept = self.act_c(predictor(concept_input))
                        concept = concept.unsqueeze(-1).expand(
                            -1, -1, self.num_monte_carlo
                        )
                        c_relaxed = torch.bernoulli(concept)
                        concept_hard = c_relaxed
                    c_prob.append(concept)
                    c_hard.append(concept_hard)
                c_prob = torch.cat([c_prob[i] for i in range(self.num_concepts)], dim=1)
                c = torch.cat([c_hard[i] for i in range(self.num_concepts)], dim=1)

            elif self.training_mode == "independent":
                # Training
                if c_true is None and concepts_train_ar is not False:
                    c_prob, c_hard = [], []
                    for c_idx, predictor in enumerate(self.concept_predictor):
                        if c_hard:
                            concept_input = torch.cat(
                                [intermediate, concepts_train_ar[:, :c_idx]], dim=1
                            )
                        else:
                            concept_input = intermediate
                        concept = self.act_c(predictor(concept_input))

                        # No Gumbel softmax because backprop can happen through the input connection
                        c_relaxed = torch.bernoulli(concept)
                        concept_hard = c_relaxed

                        # NOTE that the following train-time variables are overly good because they are taking ground truth as input. At validation time, we sample
                        c_prob.append(concept)
                        c_hard.append(concept_hard)
                    c_prob = torch.cat(
                        [c_prob[i] for i in range(self.num_concepts)], dim=1
                    )
                    c = torch.cat([c_hard[i] for i in range(self.num_concepts)], dim=1)

                else:  # Training the head with the GT concepts as input
                    c_prob = c_true.float()
                    c = c_true.float()

            else:
                raise NotImplementedError

        elif self.concept_learning == "embedding":
            # CEM
            if self.training_mode == "joint":
                # Obtaining concept embeddings
                c_p = [p(intermediate) for p in self.positive_embeddings]
                c_n = [n(intermediate) for n in self.negative_embeddings]

                # Concept probabilities from scoring function
                c_prob = [
                    self.scoring_function(torch.cat((c_p[i], c_n[i]), dim=1))
                    for i in range(self.num_concepts)
                ]

                # Final concept embedding
                z_prob = [
                    c_prob[i] * c_p[i] + (1 - c_prob[i]) * c_n[i]
                    for i in range(self.num_concepts)
                ]
                z_prob = torch.cat([z_prob[i] for i in range(self.num_concepts)], dim=1)
                c_prob = torch.cat([c_prob[i] for i in range(self.num_concepts)], dim=1)
                c = z_prob
            else:
                raise Exception("CEMs are trained jointly, change training mode")

        # Get predicted targets
        if self.concept_learning == "hard" or (
            self.concept_learning == "autoregressive" and validation
        ):
            # Hard CBM or validation of AR. Takes MCMC samples.
            # MCMC loop for predicting label
            y_pred_probs_i = 0
            for i in range(self.num_monte_carlo):
                c_i = c[:, :, i]
                y_pred_logits_i = self.head(c_i)
                if self.pred_dim == 1:
                    y_pred_probs_i += torch.sigmoid(y_pred_logits_i)
                else:
                    y_pred_probs_i += torch.softmax(y_pred_logits_i, dim=1)
            y_pred_probs = y_pred_probs_i / self.num_monte_carlo

            if self.pred_dim == 1:
                y_pred_logits = torch.logit(y_pred_probs, eps=1e-6)
            else:
                y_pred_logits = torch.log(y_pred_probs + 1e-6)

        elif self.concept_learning == "soft":
            # Soft CBM
            y_pred_logits = self.head(
                c_logit
            )  # NOTE that we're passing logits not probs in soft case as is also done by Koh et al.
            c = torch.empty_like(c_prob)

        elif self.concept_learning == "embedding" or (
            self.concept_learning == "autoregressive" and not validation
        ):
            # CEM or training of AR. Takes ground truth concepts.
            # If CEM: c are predicte embeddings, if AR: c are ground truth concepts
            y_pred_logits = self.head(c)

        return c_prob, y_pred_logits, c

    def intervene(
        self,
        concepts_interv_probs,
        concepts_mask,
        input_features,
        concepts_pred_probs,
    ):
        if self.concept_learning == "soft":
            # Soft CBM
            c_logit = torch.logit(concepts_interv_probs, eps=1e-6)
            y_pred_logits = self.head(c_logit)

        elif self.concept_learning in ("hard", "autoregressive"):
            # Hard CBM or AR
            y_pred_probs_i = 0

            if self.concept_learning == "hard":
                c_prob_mcmc = concepts_interv_probs.unsqueeze(-1).expand(
                    -1, -1, self.num_monte_carlo
                )
                c = torch.bernoulli(c_prob_mcmc)

                # Fix intervened-on concepts to ground truth
                c[concepts_mask == 1] = (
                    concepts_interv_probs[concepts_mask == 1]
                    .unsqueeze(-1)
                    .expand(-1, self.num_monte_carlo)
                )
                weight = torch.ones((c.shape[0], self.num_monte_carlo), device=c.device)

            elif self.concept_learning == "autoregressive":
                # Note: Here, concepts_interv_probs are already the hard, MCMC sampled concepts as determined by the intervene_ar function
                id = torch.nonzero(
                    concepts_interv_probs * concepts_mask == 1, as_tuple=False
                )
                weight_k = torch.log(
                    1 - concepts_pred_probs + 1e-6
                )  # If intervened-on concepts have value 0
                weight_k.index_put_(
                    list(id.t()),
                    torch.log(concepts_pred_probs + 1e-6)[id[:, 0], id[:, 1], id[:, 2]],
                    accumulate=False,
                )  # If intervened-on concepts have value 1
                weight_k = (
                    weight_k * concepts_mask
                )  # Only compute weight for intervened-on concepts
                weight = torch.sum(weight_k, dim=(1))  # Sum over concepts
                weight = torch.softmax(
                    weight, dim=-1
                )  # Replicating their implementation (from log to prob space)
                c = concepts_interv_probs

            for i in range(self.num_monte_carlo):
                c_i = c[:, :, i]
                y_pred_logits_i = self.head(c_i)
                if self.pred_dim == 1:
                    y_pred_probs_i += weight[:, i].unsqueeze(1) * torch.sigmoid(
                        y_pred_logits_i
                    )
                else:
                    y_pred_probs_i += weight[:, i].unsqueeze(1) * torch.softmax(
                        y_pred_logits_i, dim=1
                    )
            y_pred_probs = y_pred_probs_i / torch.sum(weight, dim=1).unsqueeze(1)
            if self.pred_dim == 1:
                y_pred_logits = torch.logit(y_pred_probs, eps=1e-6)
            else:
                y_pred_logits = torch.log(y_pred_probs + 1e-6)

        elif self.concept_learning == "embedding":
            # CEM
            # Get intermediate representations
            intermediate = self.encoder(input_features)
            # Obtaining concept embeddings
            c_p = [p(intermediate) for p in self.positive_embeddings]
            c_n = [n(intermediate) for n in self.negative_embeddings]
            # Final concept embedding
            z_prob = [
                concepts_interv_probs[:, i].unsqueeze(1) * c_p[i]
                + (1 - concepts_interv_probs[:, i].unsqueeze(1)) * c_n[i]
                for i in range(self.num_concepts)
            ]
            z_prob = torch.cat([z_prob[i] for i in range(self.num_concepts)], dim=1)
            y_pred_logits = self.head(z_prob)

        return y_pred_logits

    def intervene_ar(self, concepts_true, concepts_mask, input_features):
        """
        Perform an intervention on the Autoregressive CBM.

        This method performs an intervention on the Autoregressive CBM by fixing the intervened-on concepts
        to their ground-truth values and MCMC sampling the remaining concepts.
        The predicted probabilities of the intervened-on concepts are stored nevertheless to compute the reweighting.
        The reweighting is computed afterwards using the intervene function.

        Args:
            concepts_true (torch.Tensor): The ground-truth concept values. Shape: (batch_size, num_concepts, num_monte_carlo)
            concepts_mask (torch.Tensor): A mask indicating which concepts are intervened. Shape: (batch_size, num_concepts, num_monte_carlo)
            input_features (torch.Tensor): The input features for the encoder. Shape: (batch_size, input_dims)

        Returns:
            tuple: A tuple containing:
                - c_prob (torch.Tensor): Predicted concept probabilities. Shape: (batch_size, num_concepts, num_monte_carlo)
                - c (torch.Tensor): Hard predicted concept values with interventions applied. Shape: (batch_size, num_concepts, num_monte_carlo)
        """
        # Concept predictions for autoregressive model. Intervened-on concepts are fixed to ground truth
        intermediate = self.encoder(input_features)
        c_prob, c_hard = [], []
        for j, (predictor) in enumerate(self.concept_predictor):
            if c_prob:
                concept = []
                for i in range(
                    self.num_monte_carlo
                ):  # MCMC samples for evaluation and interventions, but not for joint training
                    concept_input_i = torch.cat(
                        [intermediate, torch.cat(c_hard, dim=1)[..., i]], dim=1
                    )
                    concept.append(self.act_c(predictor(concept_input_i)))
                concept = torch.cat(concept, dim=-1)
                concept_hard = torch.bernoulli(concept)[:, None, :]
                concept = concept[:, None, :]
            else:
                concept_input = intermediate
                concept = self.act_c(predictor(concept_input))
                concept = concept.unsqueeze(-1).expand(-1, -1, self.num_monte_carlo)
                concept_hard = torch.bernoulli(concept)

            concept_hard = (
                concept_hard * (1 - concepts_mask[:, j, :])[:, None, :]
                + concepts_mask[:, j, :][:, None, :]
                * concepts_true[:, j, :][:, None, :]
            )  # Only update if it is not an intervened on
            concept = (
                concept * (1 - concepts_mask[:, j, :][:, None, :])
                + concepts_mask[:, j, :][:, None, :]
                * concepts_true[:, j, :][:, None, :]
            )

            c_prob.append(concept)
            c_hard.append(concept_hard)
        c_prob = torch.cat([c_prob[i] for i in range(self.num_concepts)], dim=1)
        c = torch.cat([c_hard[i] for i in range(self.num_concepts)], dim=1)
        return c_prob, c

    def compute_temperature(self, epoch, device):
        final_temp = torch.tensor([0.5], device=device)
        init_temp = torch.tensor([1.0], device=device)
        rate = (math.log(final_temp) - math.log(init_temp)) / float(self.num_epochs)
        curr_temp = max(init_temp * math.exp(rate * epoch), final_temp)
        self.curr_temp = curr_temp
        return curr_temp

    def freeze_c(self):
        self.head.apply(freeze_module)

    def freeze_t(self):
        self.head.apply(unfreeze_module)
        self.encoder.apply(freeze_module)
        self.concept_predictor.apply(freeze_module)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x