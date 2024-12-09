from FINAL_CODE.utils.utils import set_random_seeds
from FINAL_CODE.data.synthetic_data_gen import generate_synthetic_data_leakage
from FINAL_CODE.data.data_prep import split_data, prepare_data_loaders
from FINAL_CODE.pipeline_parts.model_initial import initialize_models
from FINAL_CODE.pipeline_parts.eval import evaluate_model

def run_experiment(params, device):
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
    model_type = params.get('model_type', 'mlp')

    set_random_seeds(seed)

    # Here you can adjust noise levels by setting Sigma_x, Sigma_c, etc. if needed
    # For simplicity, we're using defaults (identity). 
    X, c, c_hat, y = generate_synthetic_data_leakage(n, d, k, J, b, l, seed=seed)
    y_zero_based = y - 1

    (X_train, X_val, X_test,
     c_train, c_val, c_test,
     c_hat_train, c_hat_val, c_hat_test,
     y_train, y_val, y_test) = split_data(X, c, c_hat, y_zero_based, train_size, val_size, test_size, seed)

    (train_loader_gb, val_loader_gb, test_loader_gb,
     train_loader_ga, val_loader_ga, test_loader_ga) = prepare_data_loaders(
        c_train, c_val, c_test,
        c_hat_train, c_hat_val, c_hat_test,
        y_train, y_val, y_test,
        batch_size, seed, device
    )

    model_gb, model_ga, criterion = initialize_models(c_train, c_hat_train, J, seed, device, model_type)

    # If model_type is MLP, we use num_epochs. For non-MLP, it can be None or ignored.
    epoch_param = num_epochs if model_type == 'mlp' else None
    model_gb.fit(train_loader_gb, num_epochs=epoch_param)
    model_ga.fit(train_loader_ga, num_epochs=epoch_param)

    model_gb.calibrate(val_loader_gb)
    model_ga.calibrate(val_loader_ga)

    test_loss_gb, test_acc_gb, avg_nll_gb = evaluate_model(model_gb, test_loader_gb, criterion, device)
    test_loss_ga, test_acc_ga, avg_nll_ga = evaluate_model(model_ga, test_loader_ga, criterion, device)

    leakage_estimate = avg_nll_gb - avg_nll_ga

    result = params.copy()
    result['test_loss_gb'] = test_loss_gb
    result['test_acc_gb'] = test_acc_gb
    result['avg_nll_gb'] = avg_nll_gb
    result['test_loss_ga'] = test_loss_ga
    result['test_acc_ga'] = test_acc_ga
    result['avg_nll_ga'] = avg_nll_ga
    result['leakage_estimate'] = leakage_estimate

    return result