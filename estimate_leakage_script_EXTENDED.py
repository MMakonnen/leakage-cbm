import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

# Import your custom modules (assumed to be provided)
from data.synthetic_dataset import generate_synthetic_data_leakage
from models.mlp import SimpleMLP
from utils import set_random_seeds
from calibration.calibration import TemperatureScaler, calibrate_model
from data.data_prep import split_data, prepare_data_loaders
from train_and_eval.train_eval import train_model_generic, evaluate_model


def main():
    # Centralized Parameters
    global_params = {
        'num_simulations': 50,  # Number of simulation runs to average results over
        'train_size': 0.6,      # Proportion of data to use for training
        'val_size': 0.2,        # Proportion of data to use for validation/calibration
        'test_size': 0.2,       # Proportion of data to use for testing
    }

    # Central Seed for Reproducibility
    base_seed = 111  # You can set this to any integer
    seed_rng = random.Random(base_seed)  # Random generator for seeds

    # Device setting (set it here centrally)
    # You can choose 'cuda', 'cpu', or 'mps' depending on your setup
    # For example, to use MPS on Apple Silicon:
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # To use CUDA if available:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # To force CPU usage:
    # device = torch.device('cpu')

    print(f"Using device: {device}")

    # Define a list of parameter configurations to test
    param_grid = [
        {
            # Synthetic data parameters
            'n': 1000,        # Number of observations/samples
            'd': 100,         # Total number of features in the dataset
            'k': 20,          # Number of concepts
            'J': 5,           # Number of target classes
            'b': 50,          # Number of features used in ground truth concepts (must satisfy k < b < d - k - l)
            'l': 5,           # Number of features excluded from leakage (controls leakage amount)
            # Model training parameters
            'batch_size': 64, # Batch size for training
            'num_epochs': 20, # Number of epochs to train the models
            # Include global parameters to be tracked
            **global_params,
        },
        # Add more configurations as needed
    ]

    # Initialize an empty list to collect results
    results = []

    for params in param_grid:
        # Sanity check to ensure data proportions sum to 1
        total_size = params['train_size'] + params['val_size'] + params['test_size']
        assert abs(total_size - 1.0) < 1e-6, "train_size, val_size, and test_size must sum to 1."

        print(f"\nRunning experiments with parameters: {params}")
        # Initialize a list to collect results for each simulation run
        simulation_results = []
        seeds = []  # Collect seeds used in all simulation runs
        for sim_run in range(params['num_simulations']):
            print(f"  Simulation run {sim_run + 1}/{params['num_simulations']}")
            # Copy params to avoid modifying the original dict
            params_run = params.copy()
            # Generate a new seed for this simulation run using the seed_rng
            seed = seed_rng.randint(0, 1_000_000)
            params_run['seed'] = seed
            seeds.append(seed)
            # Run the experiment with the given parameters
            result = run_experiment(params_run, device)
            # Append the result to simulation_results
            simulation_results.append(result)
        # Now, average the results over the simulation runs
        averaged_result = average_results(simulation_results)
        # Include the list of seeds used
        averaged_result['seeds'] = seeds
        # Append the averaged result to results
        results.append(averaged_result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    # Save or display the DataFrame
    print("\nFinal Averaged Results:")
    print(results_df)
    
    # Optionally save to CSV
    # results_df.to_csv('averaged_results.csv', index=False)


def run_experiment(params, device):
    # Unpack parameters
    seed = params['seed']
    n = params['n']
    d = params['d']
    k = params['k']
    J = params['J']
    b = params['b']
    l = params['l']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    train_size = params['train_size']
    val_size = params['val_size']
    test_size = params['test_size']

    # Set random seeds
    set_random_seeds(seed)

    # Generate synthetic data
    X, c, c_hat, y = generate_synthetic_data_leakage(n, d, k, J, b, l, seed=seed)
    y_zero_based = y - 1  # Convert labels to zero-based indexing for PyTorch

    # Split data
    (
        X_train,
        X_val,
        X_test,
        c_train,
        c_val,
        c_test,
        c_hat_train,
        c_hat_val,
        c_hat_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(X, c, c_hat, y_zero_based, train_size, val_size, test_size, seed)

    # Prepare data loaders
    (
        train_loader_gb,
        val_loader_gb,
        test_loader_gb,
        train_loader_ga,
        val_loader_ga,
        test_loader_ga,
    ) = prepare_data_loaders(
        c_train,
        c_val,
        c_test,
        c_hat_train,
        c_hat_val,
        c_hat_test,
        y_train,
        y_val,
        y_test,
        batch_size,
        seed,
        device,
    )

    # Initialize models
    (
        model_gb,
        model_ga,
        optimizer_gb,
        optimizer_ga,
        criterion,
    ) = initialize_models(c_train, c_hat_train, J, seed, device)

    # Train models
    train_model_generic(model_gb, optimizer_gb, criterion, train_loader_gb, num_epochs, device)
    train_model_generic(model_ga, optimizer_ga, criterion, train_loader_ga, num_epochs, device)

    # Calibrate models
    temperature_scaler_gb = calibrate_model(model_gb, val_loader_gb, criterion, device)
    temperature_scaler_ga = calibrate_model(model_ga, val_loader_ga, criterion, device)

    # Evaluate models
    test_loss_gb, test_acc_gb, avg_nll_gb = evaluate_model(
        model_gb, temperature_scaler_gb, criterion, test_loader_gb, device
    )
    test_loss_ga, test_acc_ga, avg_nll_ga = evaluate_model(
        model_ga, temperature_scaler_ga, criterion, test_loader_ga, device
    )

    # Compute leakage estimate
    leakage_estimate = avg_nll_gb - avg_nll_ga

    # Collect results
    result = params.copy()
    # Include only metrics in the result
    result['test_loss_gb'] = test_loss_gb
    result['test_acc_gb'] = test_acc_gb
    result['avg_nll_gb'] = avg_nll_gb
    result['test_loss_ga'] = test_loss_ga
    result['test_acc_ga'] = test_acc_ga
    result['avg_nll_ga'] = avg_nll_ga
    result['leakage_estimate'] = leakage_estimate

    return result


def average_results(simulation_results):
    # Parameters that are consistent across runs (excluding 'seed')
    param_keys = [
        'n', 'd', 'k', 'J', 'b', 'l',
        'batch_size', 'num_epochs',
        'train_size', 'val_size', 'test_size',
        'num_simulations'
    ]
    params = {k: simulation_results[0][k] for k in param_keys}
    # Metrics to average
    metrics_keys = [k for k in simulation_results[0] if k not in params]
    averaged_metrics = {}
    for key in metrics_keys:
        values = [res[key] for res in simulation_results]
        averaged_metrics[key] = np.mean(values)
    # Combine parameters and averaged_metrics
    averaged_result = params.copy()
    averaged_result.update(averaged_metrics)
    return averaged_result




def initialize_models(c_train, c_hat_train, J, seed, device):
    torch.manual_seed(seed)
    input_dim_gb = c_train.shape[1]
    input_dim_ga = c_train.shape[1] + c_hat_train.shape[1]  # Concatenated input

    model_gb = SimpleMLP(input_dim=input_dim_gb, num_classes=J).to(device)
    model_ga = SimpleMLP(input_dim=input_dim_ga, num_classes=J).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer_gb = optim.Adam(model_gb.parameters(), lr=0.001)
    optimizer_ga = optim.Adam(model_ga.parameters(), lr=0.001)

    return model_gb, model_ga, optimizer_gb, optimizer_ga, criterion



# Now, finally, call main()
if __name__ == "__main__":
    main()
