import copy
import numpy as np

def gen_experiment_par_grid():
    """
    Generate a parameter grid for the stage 1 experiments.
    In stage 1:
    - b is varied across 15 equally spaced values from 0 to d.
    - l = 0 (no additional complexity).
    
    We cross-combine:
    - dataset sizes (n)
    - noise levels
    - feature ratio (d relative to n)
    - concept ratio (k relative to d)
    - model types
    - b-values (linspace from 0 to d), l=0
    
    Returns
    -------
    param_grid : list of dict
        A list of parameter dictionaries, each specifying a configuration.
    """
    
    # Fixed parameters
    base_params = {
        'J': 5,
        'num_simulations': 50,
        'train_size': 0.7,
        'val_size': 0.15,
        'test_size': 0.15,
        'num_epochs': 20,
        'batch_size': 64
    }

    # Variable parameters
    data_sizes = [500, 2000, 10000]  # dataset sizes
    noise_levels = [0.5, 2.0]        # diagonal noise constants
    feature_ratios = [1/10, 1/50]    # d = feature_ratio * n
    concept_ratios = [1/2, 1/5]      # k = concept_ratio * d
    model_types = ['mlp', 'random_forest', 'xgboost']

    # Leakage variation: 
    # For stage 1: 
    # b ranges from 0 to d in 15 steps, l=0 only
    num_b_steps = 15

    param_grid = []

    for n in data_sizes:
        for noise in noise_levels:
            for feature_ratio in feature_ratios:
                d = int(round(n * feature_ratio))
                if d <= 0:
                    continue

                for concept_ratio in concept_ratios:
                    k = int(round(d * concept_ratio))
                    # Ensure k is valid
                    if k <= 0 or k >= d:
                        continue
                    
                    for model_type in model_types:
                        b_values = np.linspace(0, d, num_b_steps, dtype=int)
                        l_val = 0  # fixed for stage 1
                        
                        for b_val in b_values:
                            exp_params = copy.deepcopy(base_params)
                            exp_params.update({
                                'n': n,
                                'd': d,
                                'k': k,
                                'b': b_val,
                                'l': l_val,
                                'model_type': model_type,
                                'noise_level': noise
                            })
                            param_grid.append(exp_params)

    return param_grid