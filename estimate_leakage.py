
import random
import torch
import pandas as pd

from pipeline_parts.gen_experiment_grid import gen_experiment_par_grid
from pipeline_parts.experiment_runner import run_experiment
from pipeline_parts.eval import average_results


def main():
    
    # wandb.init(project='my_leakage_estimation_project')
    
    param_grid = gen_experiment_par_grid()
    results = []
    base_seed = 42
    seed_rng = random.Random(base_seed)
    device = torch.device('cpu') # or else 'cuda' (torch.cuda.is_available()) or 'mps' (torch.mps.is_available())

    for params in param_grid:
        total_size = params['train_size'] + params['val_size'] + params['test_size']
        assert abs(total_size - 1.0) < 1e-6
        simulation_results = []
        for sim_run in range(params['num_simulations']):
            params_run = params.copy()
            seed = seed_rng.randint(0, 1_000_000)
            params_run['seed'] = seed
            result = run_experiment(params_run, device)
            simulation_results.append(result)

            # # Log each simulation result
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

        # # Log averaged results
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
    # wandb.log({"final_results": wandb.Table(dataframe=results_df)})
    # wandb.finish()

if __name__ == "__main__":
    main()