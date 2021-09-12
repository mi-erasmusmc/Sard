import pathlib
import sys

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from pyreadr import pyreadr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.utils.data import Subset, DataLoader

from datasets import VisitSequenceWithLabelDataset, pad, DistillDataset, distill_pad, RETAIN_dataset, GraphDataset, \
    graph_collate
from models.Distiller import Distiller
from models.Estimator import Estimator
from models.RETAIN import RETAIN
from models.SARD_estimator import SARDLoss
from models.Transformer import Transformer
from models.VariationalGNN import VariationalGNN, VAELoss
from utils.utils import create_sequence_data, load_data, extract_best_model

DEVICE = 'cuda:' + sys.argv[1]


# DEVICE = 'cuda:1'


def get_dataset(task, model):
    """
    Gets a dataset for the given task and model

    Parameters
    ----------
    task : str      The chosen task. Data needs to have been saved in './data/task/
    model : str     The chosen model.

    Returns
    -------
    dataset :       Pytorch dataset to use
    pad_fn :        Which collate function to use with the dataset.
    population :    The plp population dataframe so we can use same indices for CV split.

    """
    from_cache = True
    data_path = pathlib.Path(f'./data/{task}/')
    population = pd.read_pickle(data_path.joinpath('population.pkl'))
    if model != 'GNN':
        if from_cache:
            covariates = pd.read_pickle(data_path.joinpath('covariates.pkl'))
            nonTemporalData = pd.read_pickle(data_path.joinpath('nonTemporal.pkl'))
            feature_map = pd.read_pickle(data_path.joinpath('map.pkl'))
        else:
            plp_data = load_data(data_path, 'python_data')
            population = plp_data['population']
            covariates = plp_data['covariates']
            nonTemporalData = plp_data['nonTemporalData']
            feature_map = plp_data['map']
        # fix r to python index conversion
        covariates.rowIdPython = covariates.rowIdPython - 1
        covariates.timeId = covariates.timeId - 1
        covariates.covariateId = covariates.covariateId - 1
        feature_map['newCovariateId'] = feature_map['newCovariateId'] - 1  # r to py
        num_features = covariates['covariateId'].max() + 1

    # normalize timeId
    if model == 'RETAIN':
        covariates.timeId = covariates.timeId / covariates.timeId.max()
    labels = population['outcomeCount'].values

    if model != 'GNN':
        if not from_cache:
            sequences, visits = create_sequence_data(covariates)
            np.savez(f'./data/{task}/{task}_not_normalized.npz', sequences, visits)
        else:
            npz_file = np.load(f'./data/{task}/{task}_not_normalized.npz', allow_pickle=True)
            sequences, visits = list(npz_file['arr_0']), list(npz_file['arr_1'])

    whole_train_indices = population[population['index'] > 0]['rowIdPython'].values - 1  # -1 because of r to py
    if model == 'SARD':
        prediction_path = f'./data/{task}/prediction.rds'  # path to predictions from LASSO
        predictions = pyreadr.read_r(prediction_path)[None]
        predictions = predictions[['rowId', 'value']]
        predictions = population.join(predictions.set_index('rowId'), on='rowId').value
        dataset = DistillDataset(linear_predictions=predictions, distill=True, seqs=sequences, labels=labels,
                                 num_features=num_features,
                                 non_temporal_data=nonTemporalData, visits=visits,
                                 train_indices=whole_train_indices, reverse=False)
        pad_fn = distill_pad
    elif model == 'RETAIN':
        dataset = RETAIN_dataset(seqs=sequences, labels=labels, num_features=num_features,
                                 non_temporal_data=nonTemporalData, visits=visits,
                                 train_indices=whole_train_indices, reverse=False)
        pad_fn = pad
    elif model == 'GNN':
        sequences_path = pathlib.Path(f'/data/home/efridgeirsson/projects/dementia/data/sequence_{task}')
        dataset = GraphDataset(root=sequences_path)

        pad_fn = graph_collate
    else:
        dataset = VisitSequenceWithLabelDataset(seqs=sequences, labels=labels, num_features=num_features,
                                                non_temporal_data=nonTemporalData, visits=visits,
                                                train_indices=whole_train_indices, reverse=False)
        pad_fn = pad

    return dataset, pad_fn, population


def objective(trial, task, model_name):
    """
    The objective function for the optuna optimization. Fits the model with the selected hyperparameters and returns
    the auc.

    Parameters
    ----------
    trial :             Optuna trial object
    task : str          What task I'm doing
    model : str         Which model I'm using

    Returns
    -------

    roc_auc :           Performance of model

    """

    dataset, pad_fn, population = get_dataset(task, model_name)
    model_parameters, fit_parameters = get_hyperparameters(trial, model_name)

    model_parameters['num_features'] = dataset.num_features
    if model_name != 'GNN':
        model_parameters['max_visits'] = dataset.max_visits
    elif model_name == 'SARD':
        distill_learning_rate = fit_parameters['distill_learning_rate']
        finetune_learning_rate = fit_parameters['finetune_learning_rate']
        alpha = model_parameters['alpha']
    else:
        lr = fit_parameters['lr']
    weight_decay = fit_parameters['weight_decay']

    batch_size = 8
    fit_parameters = {'epochs': 30, 'lr': 2.5e-5, 'l2': weight_decay,
                      'prefix': task}
    results_dir = pathlib.Path().cwd().joinpath(
        f'SavedModels/{task}/{model_name}/grid_search/{trial.number}/')

    # use one fold as validation and rest for training
    train_indices = population[population['index'].isin([1, 2])]['rowIdPython'].values - 1
    val_indices = population[population['index'] == 3]['rowIdPython'].values - 1
    train_dataset = Subset(dataset=dataset, indices=train_indices)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  pin_memory=False, num_workers=0, collate_fn=pad_fn)
    val_dataset = Subset(dataset=dataset, indices=val_indices)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                pin_memory=False, num_workers=0, collate_fn=pad_fn)

    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    fit_parameters['results_dir'] = results_dir
    labels = population['outcomeCount'].values
    whole_train_indices = population[population['index'] > 0]['rowIdPython'].values - 1  # -1 because of r to py
    y_train = labels[whole_train_indices]
    pos_weight = torch.as_tensor((y_train.shape[0] - y_train.sum()) / y_train.sum(), dtype=torch.float32)
    if model_name == 'SARD':
        distill_loss = SARDLoss(distill_loss=nn.BCEWithLogitsLoss(pos_weight), distill=True)
        finetune_loss = SARDLoss(distill_loss=nn.BCEWithLogitsLoss(pos_weight),
                                 classifier_loss=nn.BCEWithLogitsLoss(pos_weight), distill=False, alpha=alpha)
        model = Distiller(estimator=Estimator, model=Transformer, model_parameters=model_parameters,
                          fit_parameters=fit_parameters, device=DEVICE)
        model.distill(distill_loss, train_dataloader, val_dataloader, distill_learning_rate)

        model.finetune(finetune_loss, train_dataloader, val_dataloader, finetune_learning_rate)
    else:
        if model_name == 'RETAIN':
            base_model = RETAIN
            model_parameters['dim_num'] = dataset.num.shape[1] + 1
        elif model_name == 'Transformer':
            base_model = Transformer
        elif model_name == 'GNN':
            base_model = VariationalGNN
            alpha = trial.suggest_discrete_uniform('alpha', 0, 1, 0.1)
            criterion = VAELoss(alpha=alpha, pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        model = Estimator(model=base_model, model_parameters=model_parameters, fit_parameters=fit_parameters,
                          device=DEVICE, criterion=criterion)
        # lr_finder = LRFinder(model=model.model, optimizer=model.optimizer, criterion=model.criterion)
        # lr_finder.range_test(train_dataloader, start_lr=1e-8, end_lr=1e-1, num_iter=100)
        model.fit(train_dataloader, val_dataloader)

    preds = model.predict_proba(val_dataloader)
    y_val = val_dataset.dataset.labels[val_indices]
    roc_auc = roc_auc_score(y_val, preds)

    return roc_auc


def random_search(database_name=None, task='readmission', model_name='RETAIN'):
    """
    Performs hyperparameter optimization using 100 random samples

    Parameters
    ----------
    database_name : Name of sqlite database which stores info about grid search
    task :          Which task to perform
    model :         Which model to use

    Returns
    -------
    best_params : Best parameters found with random search

    """

    study = optuna.create_study(storage=f'sqlite:///{database_name}.db',
                                study_name=f'{database_name}',
                                load_if_exists=True, direction='maximize')
    study.optimize(lambda trial: objective(trial, task, model_name), n_trials=100,
                   catch=(RuntimeError,))  # doesn't stop if I get a cuda out of memory error
    best_params = study.best_params
    return best_params


def get_hyperparameters(trial, model):
    """
    Function to selecter hyperparameter space to optimize over depending on the model

    Parameters
    ----------
    trial : A optuna trial instance
    model : str The chosen model

    Returns
    -------
    model_parameters : a dict with parameters for the model
    fit_parameters : a dict with model agnostic parameters to tune.

    """
    model_parameters = {}
    fit_parameters = {}
    if model == 'RETAIN':
        model_parameters['dim_emb'] = trial.suggest_int('dim_emb', 32, 256, 32)
        model_parameters['dropout_emb'] = trial.suggest_uniform('dropout_emb', 0, 0.3)
        model_parameters['dim_alpha'] = trial.suggest_int('dim_alpha', 32, 256, 32)
        model_parameters['dim_beta'] = trial.suggest_int('dim_beta', 32, 256, 32)
        model_parameters['dropout_context'] = trial.suggest_uniform('dropout_context', 0, 0.3)
        model_parameters['num_layers'] = trial.suggest_int('num_layers', 1, 8, 1)
        fit_parameters['lr'] = 3e-4
        fit_parameters['weight_decay'] = 1e-4
    elif model == 'Transformer':
        embedding_per_head = trial.suggest_int('embedding_per_head', 16, 96, 16)
        model_parameters['num_heads'] = trial.suggest_int('num_heads', 2, 6, 2)
        model_parameters['embedding_dim'] = (model_parameters[
                                                 'num_heads'] * embedding_per_head) - 4  # -4 numerical variables
        model_parameters['num_layers'] = trial.suggest_int('attn_depth', 1, 6, 1)
        model_parameters['num_hidden'] = trial.suggest_int('num_hidden', 256, 2048, 1)
        model_parameters['attention_dropout'] = trial.suggest_discrete_uniform('attention_dropout', 0, 0.3, 0.05)
        model_parameters['residual_dropout'] = trial.suggest_discrete_uniform('residual_dropout', 0, 0.3, 0.05)
        model_parameters['ffn_dropout'] = trial.suggest_discrete_uniform('ffn_dropout', 0, 0.3, 0.05)
        model_parameters['max_len'] = 365
        model_parameters['parallel_pools'] = 10
    elif model == 'SARD':
        fit_parameters['finetune_learning_rate'] = 1.5e-4
        fit_parameters['distill_learning_rate'] = 3e-4
        model_parameters['alpha'] = trial.suggest_discrete_uniform('alpha', 0, 0.3, 0.05)
        model_parameters['num_heads'] = 4
        model_parameters['embedding_dim'] = 124
        model_parameters['num_layers'] = 4
        model_parameters['num_hidden'] = 1505
        model_parameters['attention_dropout'] = 0.2
        model_parameters['residual_dropout'] = 0.2
        model_parameters['ffn_dropout'] = 0.1
        model_parameters['max_len'] = 365
        model_parameters['parallel_pools'] = 10
    elif model == 'GNN':
        model_parameters['none_graph_features'] = 4
        model_parameters['num_layers'] = trial.suggest_int('num_layers', 1, 6, 1)
        model_parameters['num_heads'] = trial.suggest_int('num_heads', 1, 4, 1)
        model_parameters['dim_embedding'] = trial.suggest_int('dim_embedding', 32, 1024, 1)
        model_parameters['dropout'] = trial.suggest_uniform('dropout', 0, 0.3)
        model_parameters['attention_dropout'] = trial.suggest_uniform('attention_dropout', 0, 0.3)
    fit_parameters['lr'] = 5e-5
    fit_parameters['weight_decay'] = 1e-4

    return model_parameters, fit_parameters


def test_model(task, model_name):
    """
    Function I use to test a given model once I have found the best hyperparameters. I refit the model
    on the whole training set and test on the test set.

    Parameters
    ----------
    task : str          The chosen task.
    model_name : str    The model to use.

    Returns
    -------

    """
    dataset, pad_fn, population = get_dataset(task, model_name)
    labels = population['outcomeCount'].values
    whole_train_indices = population[population['index'] > 0]['rowIdPython'].values - 1  # -1 because of r to py

    # load study object, find index of trial with best performance
    study = optuna.load_study(study_name=f'grid_search_{model_name}_{task}',
                              storage=f'sqlite:///grid_search_{model_name}_{task}.db')
    best_trial_id = study.best_trial.number

    grid_search_directory = pathlib.Path().cwd().joinpath(
        f'SavedModels/{task}/{model_name}/grid_search/{best_trial_id}')
    best_model_file = extract_best_model(grid_search_directory)
    best_model = torch.load(best_model_file, map_location='cpu')
    model_parameters = best_model['model_hyperparameters']

    batch_size = 512
    whole_train_dataset = Subset(dataset, whole_train_indices)
    whole_train_dataloader = DataLoader(whole_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False,
                                        num_workers=0, collate_fn=pad_fn)
    train_indices = population[population['index'].isin([1, 2])]['rowIdPython'].values - 1
    val_indices = population[population['index'] == 3]['rowIdPython'].values - 1
    train_dataset = Subset(dataset=dataset, indices=train_indices)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  pin_memory=False, num_workers=0, collate_fn=pad_fn)
    val_dataset = Subset(dataset=dataset, indices=val_indices)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                pin_memory=False, num_workers=0, collate_fn=pad_fn)

    lr = 3e-4
    weight_decay = 1e-4

    fit_parameters = {'epochs': 30, 'lr': lr, 'l2': weight_decay,
                      'prefix': task}
    results_dir = pathlib.Path().cwd().joinpath(
        f'SavedModels/{task}/{model_name}/best_params_catboost_distillation/')
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    fit_parameters['results_dir'] = results_dir
    y_train = labels[whole_train_indices]
    pos_weight = torch.as_tensor((y_train.shape[0] - y_train.sum()) / y_train.sum(), dtype=torch.float32)

    if model_name == 'SARD':
        distill_loss = SARDLoss(distill_loss=nn.BCEWithLogitsLoss(pos_weight), distill=True)
        finetune_loss = SARDLoss(distill_loss=nn.BCEWithLogitsLoss(pos_weight),
                                 classifier_loss=nn.BCEWithLogitsLoss(pos_weight), distill=False, alpha=0.2)
        model = Distiller(estimator=Estimator, model=Transformer, model_parameters=model_parameters,
                          fit_parameters=fit_parameters, device=DEVICE)

        model.distill(distill_loss, train_dataloader, val_dataloader, fit_parameters['lr'])
        model.finetune(finetune_loss, train_dataloader, val_dataloader, 5e-4)

        # TODO get learning rates in an automatic manner.
        data = torch.load(
            '/data/home/efridgeirsson/PycharmProjects/Sard/SavedModels/mortality/SARD/best_params_2/distill/mortality_epochs:16_auc:0.9347_val_loss:3.3998')
        learning_rates = data['learning_rates']
        model.distill_whole_training_set(distill_loss, whole_train_dataloader, 3e-4, epochs=16,
                                         learning_rates=learning_rates)
        model.finetune_whole_training_set(finetune_loss, whole_train_dataloader, 2e-6, epochs=1,
                                          distill_epoch=15, learning_rates=[2e-6])
    else:
        if model_name == 'Transformer':
            base_model = Transformer
        if model_name == 'RETAIN':
            base_model = RETAIN
            model_parameters['dim_num'] = dataset.num.shape[1] + 1
        if model_name == 'GNN':
            base_model = VariationalGNN
            criterion = VAELoss(pos_weight=pos_weight, alpha=study.best_params['alpha'])
    model = Estimator(model=base_model, model_parameters=model_parameters, fit_parameters=fit_parameters,
                      device='cuda:1', criterion=criterion)
    # lr_finder = LRFinder(model=model.model, optimizer=model.optimizer, criterion=model.criterion)
    # lr_finder.range_test(whole_train_dataloader, start_lr=1e-8, end_lr=1e-1, num_iter=100)
    learning_rates = best_model['learning_rates']
    epochs = best_model['epoch']
    model.fit_whole_training_set(whole_train_dataloader, epochs=epochs, learning_rates=learning_rates)

    test_indices = population[population['index'] < 0]['rowIdPython'].values - 1  # -1 because of r to py
    test_dataset = Subset(dataset, test_indices)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_fn)

    preds = model.predict_proba(test_dataloader)
    y_test = test_dataset.dataset.labels[test_indices]
    roc_auc = roc_auc_score(y_test, preds)

    precision, recall, _ = precision_recall_curve(y_test, preds)
    auprc = auc(recall, precision)

    np.save(f'./results/{task}/{model_name}_predictions.npy', preds)

    # TODO save roc_auc and auprc


if __name__ == '__main__':
    model_name = 'GNN'
    task = 'readmission'

    best_params = random_search(database_name=f'grid_search_{model_name}_{task}', model_name=model_name, task=task)
