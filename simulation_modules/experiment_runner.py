import numpy as np

from utils.utils import set_random_seeds
from data.synthetic_data_gen import generate_synthetic_data_leakage
from data.data_prep import split_data, prepare_data_loaders
from models.model_initial import initialize_models
from simulation_modules.eval import evaluate_model

def run_experiment(params, device, max_tries=10):
    """
    Runs a single experiment with the given parameters and returns the results.

    Parameters:
    - params: Dictionary of experimental parameters.
    - device: Computation device ('cpu' or 'cuda').
    - max_tries: Maximum attempts to generate a valid dataset.

    Returns:
    - result: Dictionary containing evaluation metrics and leakage estimate.
    """
    # Extract parameters
    seed = params['seed']
    n, d, k, J, b, l = params['n'], params['d'], params['k'], params['J'], params['b'], params['l']
    batch_size, num_epochs = params['batch_size'], params['num_epochs']
    train_size, val_size, test_size = params['train_size'], params['val_size'], params['test_size']
    model_type = params.get('model_type', 'mlp')
    noise_level = params['noise_level']

    # Set random seed for reproducibility
    set_random_seeds(seed)

    # Define covariance matrices for noise
    sigma_x = noise_level * np.eye(d)
    sigma_c = noise_level * np.eye(k)
    sigma_c_hat = noise_level * np.eye(k)
    sigma_y = noise_level * np.eye(J)

    # Attempt to generate a dataset with all classes in the training set
    for attempt in range(max_tries):
        # Generate synthetic data
        X, c, c_hat, y = generate_synthetic_data_leakage(
            n=n,
            d=d,
            k=k,
            J=J,
            b=b,
            l=l,
            sigma_x=sigma_x,
            sigma_c=sigma_c,
            sigma_c_hat=sigma_c_hat,
            sigma_y=sigma_y,
            seed=seed
        )

        # Split data into train, validation, and test sets
        (X_train, X_val, X_test,
         c_train, c_val, c_test,
         c_hat_train, c_hat_val, c_hat_test,
         y_train, y_val, y_test) = split_data(X, c, c_hat, y_zero_based, train_size, val_size, test_size, seed)

        # Check if all classes are represented in the training set
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique_train, counts_train))
        all_classes_present = all(c in class_counts for c in range(J))

        if all_classes_present:
            break  # Valid dataset generated
        else:
            # Update seed and try again
            seed += 1
            set_random_seeds(seed)
    else:
        # If attempts fail, skip this configuration
        print(f"Skipping configuration {params} as we couldn't get all classes in the training set after {max_tries} tries.")
        return None

    # Prepare DataLoaders
    (train_loader_gb, val_loader_gb, test_loader_gb,
     train_loader_ga, val_loader_ga, test_loader_ga) = prepare_data_loaders(
        c_train, c_val, c_test,
        c_hat_train, c_hat_val, c_hat_test,
        y_train, y_val, y_test,
        batch_size, seed, device
    )

    # Initialize models and loss function
    model_gb, model_ga, criterion = initialize_models(c_train, c_hat_train, J, seed, device, model_type)

    # Train models
    epoch_param = num_epochs if model_type == 'mlp' else None
    model_gb.fit(train_loader_gb, num_epochs=epoch_param)
    model_ga.fit(train_loader_ga, num_epochs=epoch_param)

    # Calibrate models
    model_gb.calibrate(val_loader_gb)
    model_ga.calibrate(val_loader_ga)

    # Evaluate models
    test_loss_gb, test_acc_gb, avg_nll_gb = evaluate_model(model_gb, test_loader_gb, criterion, device)
    test_loss_ga, test_acc_ga, avg_nll_ga = evaluate_model(model_ga, test_loader_ga, criterion, device)

    # Compute leakage estimate
    leakage_estimate = avg_nll_gb - avg_nll_ga

    # Compile results
    result = params.copy()
    result.update({
        'test_loss_gb': test_loss_gb,
        'test_acc_gb': test_acc_gb,
        'avg_nll_gb': avg_nll_gb,
        'test_loss_ga': test_loss_ga,
        'test_acc_ga': test_acc_ga,
        'avg_nll_ga': avg_nll_ga,
        'leakage_estimate': leakage_estimate
    })

    return result
