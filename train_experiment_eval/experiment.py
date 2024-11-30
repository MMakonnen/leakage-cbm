# custom modules
from other.utils import set_random_seeds
from data.synthetic_dataset import generate_synthetic_data_leakage
from data.data_prep import split_data, prepare_data_loaders
from models.mlp import initialize_mlps
from train_experiment_eval.train import train_model_generic
from calibration.calibration import calibrate_model
from train_experiment_eval.eval import evaluate_model


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
    ) = initialize_mlps(c_train, c_hat_train, J, seed, device)

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