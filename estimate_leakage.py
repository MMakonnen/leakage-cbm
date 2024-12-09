import random
import torch
import pandas as pd
from tqdm import tqdm
# import wandb

from pipeline_parts.gen_experiment_grid import gen_experiment_par_grid
from pipeline_parts.experiment_runner import run_experiment
from pipeline_parts.eval import average_results

def main():

    # # Initialize W&B run
    # wandb.init(project='my_leakage_estimation_project', config={
    #     "description": "Experiment with progress bars and W&B logging"
    # })
    
    param_grid = gen_experiment_par_grid()
    results = []
    base_seed = 42
    seed_rng = random.Random(base_seed)
    device = torch.device('cpu') # or else 'cuda' (torch.cuda.is_available()) or 'mps' (torch.mps.is_available())

    print(f"Number of configurations: {len(param_grid)}")
    # Wrap param_grid in tqdm to track progress over configurations
    for config_idx, params in enumerate(tqdm(param_grid, desc="Configurations")):
        total_size = params['train_size'] + params['val_size'] + params['test_size']
        assert abs(total_size - 1.0) < 1e-6

        simulation_results = []

        # Progress bar for simulations per configuration
        num_sims = params['num_simulations']
        for sim_run in tqdm(range(num_sims), desc=f"Simulations for config {config_idx+1}/{len(param_grid)}", leave=False):
            params_run = params.copy()
            seed = seed_rng.randint(0, 1_000_000)
            params_run['seed'] = seed

            result = run_experiment(params_run, device)
            simulation_results.append(result)

            # # Log each simulation result to W&B
            # wandb.log({
            #     "model_type": params['model_type'],
            #     "n": params['n'],
            #     "d": params['d'],
            #     "k": params['k'],
            #     "b": params['b'],
            #     "l": params['l'],
            #     "test_loss_gb": result['test_loss_gb'],
            #     "test_acc_gb": result['test_acc_gb'],
            #     "avg_nll_gb": result['avg_nll_gb'],
            #     "test_loss_ga": result['test_loss_ga'],
            #     "test_acc_ga": result['test_acc_ga'],
            #     "avg_nll_ga": result['avg_nll_ga'],
            #     "leakage_estimate": result['leakage_estimate'],
            #     "simulation_run": sim_run + 1
            # })

        averaged_result = average_results(simulation_results)
        results.append(averaged_result)

        # # Log averaged results to W&B
        # wandb.log({
        #     "model_type_avg": params['model_type'],
        #     "avg_test_loss_gb": averaged_result['test_loss_gb'],
        #     "avg_test_acc_gb": averaged_result['test_acc_gb'],
        #     "avg_nll_gb": averaged_result['avg_nll_gb'],
        #     "avg_test_loss_ga": averaged_result['test_loss_ga'],
        #     "avg_test_acc_ga": averaged_result['test_acc_ga'],
        #     "avg_nll_ga": averaged_result['avg_nll_ga'],
        #     "avg_leakage_estimate": averaged_result['leakage_estimate'],
        #     "num_simulations": averaged_result['num_simulations']
        # })

    results_df = pd.DataFrame(results)
    print("\nAll experiments completed. Final Averaged Results:")
    print(results_df)

    # # Log final results DataFrame as a W&B artifact or table
    # wandb.log({"final_results": wandb.Table(dataframe=results_df)})

    # # Finish W&B run
    # wandb.finish()

if __name__ == "__main__":
    main()