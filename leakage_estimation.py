import random
import torch
import pandas as pd

# custom modules
from train_experiment_eval.experiment import run_experiment
from train_experiment_eval.eval import average_results


def main():
    
    # Centralized Parameters
    global_params = {
        'num_simulations': 50,  # number of simulation runs to average results over
        'train_size': 0.6,      # training proportion
        'val_size': 0.2,        # validation/calibration proportion
        'test_size': 0.2,       # testing proportion
    }

    # Central Seed and random generator for Reproducibility (centrally controls all randomness)
    base_seed = 42
    seed_rng = random.Random(base_seed)
    
    # Allows for GPU computations if available (choose 'cuda', 'cpu', or 'mps')
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # mac
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # force cpu usage
    print(f"Using device: {device}")

    # List of parameter configurations to test
    param_grid = [
        {
            # Synthetic data parameters
            'n': 1000,
            'd': 100,
            'k': 20,
            'J': 5,
            'b': 50,
            'l': 5,
            # Model training parameters
            'batch_size': 64,
            'num_epochs': 20,
            # Include global parameters to be tracked
            **global_params,
        },
        {
            # Synthetic data parameters
            'n': 10000,
            'd': 100,
            'k': 20,
            'J': 5,
            'b': 50,
            'l': 5,
            # Model training parameters
            'batch_size': 64,
            'num_epochs': 20,
            # Include global parameters to be tracked
            **global_params,
        },
    ]

    # Collect average results across simulation runs per parameter configuration
    results = []

    for params in param_grid:

        # Sanity check to ensure data proportions sum to 1
        total_size = params['train_size'] + params['val_size'] + params['test_size']
        assert abs(total_size - 1.0) < 1e-6, "train_size, val_size, and test_size must sum to 1."

        print(f"\nRunning experiments with parameters: {params}")
        # Collect simulation run results for specific parameter configuration
        simulation_results = []
        for sim_run in range(params['num_simulations']):
            print(f"  Simulation run {sim_run + 1}/{params['num_simulations']}")
            # Copy params to avoid modifying the original dict
            params_run = params.copy()
            # Generate new seed for simulation run
            seed = seed_rng.randint(0, 1_000_000)
            params_run['seed'] = seed
            # Run experiment with given parameters
            result = run_experiment(params_run, device)
            simulation_results.append(result)

        # average results over simulation runs
        averaged_result = average_results(simulation_results)
        results.append(averaged_result)

    # Convert results to DataFrame and display
    results_df = pd.DataFrame(results)
    print("\nFinal Averaged Results:")
    print(results_df)


if __name__ == "__main__":
    main()

