import copy
import numpy as np

def gen_experiment_par_grid(
    data_sizes,
    noise_levels,
    feature_dimensions,
    number_of_concepts,
    model_types,
    l_values,
    base_params,
    desired_num_b_steps
):
    """
    Generates a parameter grid by combining different configurations.

    Parameters:
    - data_sizes: List of dataset sizes (number of observations).
    - noise_levels: List of diagonal noise constants.
    - feature_dimensions: List of feature space dimensions.
    - number_of_concepts: List of numbers of concepts.
    - model_types: List of model types to use.
    - l_values: List of leakage values.
    - base_params: Dictionary of fixed parameters for the experiments.
    - desired_num_b_steps: Desired number of steps for `b` in its range.

    Returns:
    - param_grid: List of dictionaries, each containing a unique configuration.
    """
    param_grid = []

    for n in data_sizes:
        for noise in noise_levels:
            for d in feature_dimensions:
                if d <= 0:  # Skip invalid dimensions
                    continue

                for k in number_of_concepts:
                    if k <= 0 or k >= d:  # Skip invalid concepts
                        continue

                    for model_type in model_types:
                        for l_val in l_values:
                            # Calculate range for b based on k and l_val
                            min_b = k + 1
                            max_b = d - k - l_val - 1

                            if min_b > max_b:  # Skip invalid b ranges
                                continue

                            # Determine valid b values
                            total_possible_b = max_b - min_b + 1
                            num_b_steps = min(desired_num_b_steps, total_possible_b)
                            b_values = np.linspace(min_b, max_b, num_b_steps, dtype=int)
                            b_values = np.unique(b_values)  # Ensure unique, sorted values

                            for b_val in b_values:
                                # Create configuration dictionary
                                exp_params = copy.deepcopy(base_params)
                                exp_params.update({
                                    'n': n,
                                    'd': d,
                                    'k': k,
                                    'l': l_val,
                                    'b': b_val,
                                    'model_type': model_type,
                                    'noise_level': noise
                                })

                                param_grid.append(exp_params)

    return param_grid
