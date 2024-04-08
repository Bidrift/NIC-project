import pickle

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
from sklearn.model_selection import KFold, GridSearchCV, train_test_split

# Dictionary mapping user-friendly names to corresponding sklearn metrics names


# Cross-validation setup
CV = KFold(n_splits=5, shuffle=True, random_state=11)

# Function to tune a model using GridSearchCV
def tune_model(model, param_grid, X_train, y_train, cv=None, scoring='neg_mean_absolute_error'):
    metric_mapper = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2",
        "msle": "neg_mean_squared_log_error"
    }
    if scoring in metric_mapper:
        scoring = metric_mapper[scoring]

    if cv is None:
        cv = CV

    searcher = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    searcher.fit(X_train, y_train)
    return searcher.best_estimator_

# Function to evaluate a tuned model
def evaluate_tuned_model(tuned_model, X_train, X_test, y_train, y_test, train=True, metrics=None):
    if metrics is None:
        metrics = ['mse']

    if isinstance(metrics, str):
        metrics = [metrics]

    if 'msle' in metrics and (y_train <= 0).any():
        metrics.remove('msle')

    if train:
        tuned_model.fit(X_train, y_train)
        
    evaluation_metrics = {
        "mse": mean_squared_error,
        "mae": mean_absolute_error,
        "r2": r2_score,
        'rmse': lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False),
        "msle": mean_squared_log_error
    }

    y_pred = tuned_model.predict(X_test)
    scores = {metric: evaluation_metrics[metric](y_test, y_pred) for metric in metrics}
    return tuned_model, scores

# Function to save a model
def save_model(tuned_model, path):
    with open(path, 'wb') as f:
        pickle.dump(tuned_model, f)

# Function to load a model
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
# Function to apply model tuning and evaluation
def apply_model(model, X_train, X_test, y_train, y_test, param_grid, save=True, save_path=None, test_size=0.2,
                tune_metric=None, test_metrics=None, cv=None):
    
    tuned_model = tune_model(model, param_grid, X_train, y_train, cv=cv, scoring=tune_metric)

    model, results = evaluate_tuned_model(tuned_model, X_train, X_test, y_train, y_test, metrics=test_metrics)

    if save:
        save_model(tuned_model, save_path)

    return model, results

def try_ridge_regression(X, y, model=Ridge, param_grid=None, save=True, save_path=None,
                         test_size=0.2, tune_metric=None, test_metrics=None, cv=None):
    if param_grid is None:
        param_grid = {"alpha": np.logspace(0.001, 10, 20)}

    return apply_model(model, X, y, param_grid, save=save, save_path=save_path,
                       test_size=test_size, tune_metric=tune_metric, test_metrics=test_metrics, cv=cv)

if __name__ == "__main__":
    # Generate synthetic data
    X, Y = make_regression(n_samples=4000, n_features=20, random_state=18, n_informative=8)

    # Try Ridge regression model
    ridge_model, evaluation_results = try_ridge_regression(X, Y, save=False, test_metrics=['mse', 'rmse', 'r2', "msle"], tune_metric='mse')

    print(evaluation_results)