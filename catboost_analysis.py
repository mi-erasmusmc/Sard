"""
Train a catboost model with grid search
"""
import json
import pathlib

import catboost
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from utils.data_utils import add_age_gender
from utils.data_utils import window_data_sorted
from utils.utils import load_data, NpEncoder

if __name__ == '__main__':
    """
    This script fits a Catboost model. First it uses randomised search to select hyperparameters. Then it takes the best
    estimator and tests it on the test data
    
    """
    task = 'mortality'
    data_folder = pathlib.Path(f'/data/home/efridgeirsson/projects/dementia/data/sequence_{task}/')

    plp_data = load_data(data_folder=data_folder, name='python_data')

    results_directory = pathlib.Path.cwd().joinpath('SavedModels', f'{task}', 'catboost', 'best_params')
    if not results_directory.exists():
        results_directory.mkdir(parents=True)
    population = plp_data["population"]
    indices = dict()
    indices['test'] = np.argwhere(population['index'].values == -1.0).squeeze()
    indices['train'] = np.argwhere(population['index'].values > 0).squeeze()

    window_lengths = (30, 180, 365)
    feature_matrix_counts, windowed_feature_names = window_data_sorted(
        window_lengths=list(window_lengths),
        feature_matrix=plp_data['data'],
        all_feature_names=plp_data['feature_names'])

    feature_matrix_counts = feature_matrix_counts.T

    feature_matrix_counts.data = np.clip(feature_matrix_counts.data, 0, 1)  # counts to binary

    feature_matrix_counts, windowed_feature_names = add_age_gender(feature_matrix_counts,
                                                                   plp_data['nonTemporalData'], windowed_feature_names,
                                                                   age_normalized=False)
    outcomes = plp_data['outcomes']
    X = feature_matrix_counts
    y = outcomes
    X_train = X[indices['train']]
    X_test = X[indices['test']]
    y_train = y[indices['train']]
    y_test = y[indices['test']]

    # custom cv generator to get exactly the same cv split as in PLP
    cv_iterator = []
    indexes = population[population['index'] > 0]['index'].unique()
    for i in indexes:
        val_idx = i
        val_indices = population[population['index'] == val_idx]['rowIdPython'].values - 1
        train_idx = indexes[indexes != i]
        train_indices = population[population['index'].isin(train_idx)]['rowIdPython'].values - 1
        cv_iterator.append((train_indices, val_indices))

    train_indices = population[population['index'].isin([1, 2])]['rowIdPython'].values - 1
    val_indices = population[population['index'] == 3]['rowIdPython'].values - 1
    X_train, y_train = feature_matrix_counts[train_indices], outcomes[train_indices]
    X_val, y_val = feature_matrix_counts[val_indices], outcomes[val_indices]

    # train the regression model over several choices of regularization parameter
    scaler = MaxAbsScaler()
    params = {'catboost__depth': list(np.arange(1, 11)),
              'catboost__loss_function': ['Logloss'],
              'catboost__iterations': [100, 250, 500, 1000, 2000, 3000, 5000, 10000],
              'catboost__learning_rate': [0.001, 0.01, 0.03, 0.1, 0.2, 0.3],
              'catboost__l2_leaf_reg': [1, 3, 5, 10, 100],
              'catboost__border_count': [5, 10, 20, 32, 50, 100, 200]}

    model = catboost.CatBoostClassifier(task_type='GPU', devices='0', auto_class_weights='Balanced',
                                        logging_level='Silent', early_stopping_rounds=30)

    # only normalize continuous features
    ct = ColumnTransformer([('scaler', MaxAbsScaler(), slice(-4, -1))], remainder='passthrough')
    pipeline = Pipeline([('scaler', ct), ('catboost', model)])

    grid_search = RandomizedSearchCV(estimator=pipeline, param_distributions=params, scoring='roc_auc', refit=True,
                                     cv=cv_iterator, n_jobs=1, verbose=10, random_state=42, n_iter=100)
    grid_search.fit(X, y)

    best_estimator = grid_search.best_estimator_

    best_params = grid_search.best_params_

    with open(results_directory.joinpath('catboost_best_params.json'), 'w+') as f:
        json.dump(best_params, f, cls=NpEncoder)
    best_parameters = {key.split('__')[1]: value for (key, value) in best_params.items()}
    best_model = catboost.CatBoostClassifier(task_type='GPU', auto_class_weights='Balanced',
                                             **best_parameters)

    X_train = ct.fit_transform(X_train)
    X_val = ct.transform(X_val)

    train_data = catboost.Pool(data=X_train.astype(np.float32), label=y_train)
    eval_data = catboost.Pool(data=X_val.astype(np.float32), label=y_val.values)
    best_model.fit(train_data, eval_set=eval_data, early_stopping_rounds=30)

    X_test = ct.transform(X_test)
    predictions_test = best_model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, predictions_test)
    precision, recall, threshold = precision_recall_curve(y_test, predictions_test)
    auprc = auc(recall, precision)

    np.save(f'./results/{task}/catboost_predictions.npy', predictions_test)

    all_predictions = best_model.predict_proba(ct.transform(X))[:, 1]

    np.save(f'./data/{task}/catboost_predictions.npy', all_predictions)
