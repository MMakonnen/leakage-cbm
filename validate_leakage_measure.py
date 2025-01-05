import random
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from simulation_modules.gen_experiment_grid import gen_experiment_par_grid
from simulation_modules.experiment_runner import run_experiment
from simulation_modules.eval import average_results


def main():
    """
    Validates the leakage measure on fully synthetic data by running experiments
    across different parameter configurations and saving the results.
    """
    # Define variable parameters
    data_sizes = [500, 2000, 10000]                     # Number of observations
    noise_levels = [0.5, 2.0]                           # Noise levels
    feature_dimensions = [500, 2500]                    # Feature space dimensions
    number_of_concepts = [50, 200]                      # Number of concepts
    model_types = ['mlp', 'random_forest', 'xgboost']   # Model types to test
    l_values = [0]                                      # Leakage control values
    desired_num_b_steps = 30                            # Granularity of leakage measurement

    # Define base parameters
    base_params = {
        'J': 5,
        'num_simulations': 5,
        'train_size': 0.7,
        'val_size': 0.15,
        'test_size': 0.15,
        'num_epochs': 20,
        'batch_size': 64
    }

    # Generate parameter grid for experiments
    param_grid = gen_experiment_par_grid(
        data_sizes=data_sizes,
        noise_levels=noise_levels,
        feature_dimensions=feature_dimensions,
        number_of_concepts=number_of_concepts,
        model_types=model_types,
        l_values=l_values,
        base_params=base_params,
        desired_num_b_steps=desired_num_b_steps
    )
    results = []
    base_seed = 42
    seed_rng = random.Random(base_seed)
    device = torch.device('cpu')  # Use 'cuda' if GPU is available or 'mps' for Apple silicon

    print(f"Number of configurations: {len(param_grid)}")

    # Iterate over each parameter configuration
    for config_idx, params in enumerate(tqdm(param_grid, desc="Configurations")):
        # Ensure data splits sum to 1
        total_size = params['train_size'] + params['val_size'] + params['test_size']
        assert abs(total_size - 1.0) < 1e-6, f"Data splits do not add up to 1 for configuration {config_idx+1}"

        simulation_results = []

        # Run multiple simulations for the current configuration
        num_sims = params['num_simulations']
        for sim_run in tqdm(range(num_sims), desc=f"Simulations for config {config_idx+1}/{len(param_grid)}", leave=False):
            params_run = params.copy()
            seed = seed_rng.randint(0, 1_000_000)  # Generate random seed for this simulation
            params_run['seed'] = seed

            result = run_experiment(params_run, device)  # Run the experiment
            simulation_results.append(result)

        # Average results across simulations for this configuration
        averaged_result = average_results(simulation_results)
        results.append(averaged_result)

    # Compile results into a DataFrame
    results_df = pd.DataFrame(results)
    print("\nAll experiments completed. Final Averaged Results:")
    print(results_df)

    # Save results to CSV with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"simulation_results/res_leak_measure_val/res_leak_measure_val_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nFinal results saved to: {results_path}")

if __name__ == "__main__":
    main()
