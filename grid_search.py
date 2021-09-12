import pathlib
import sys

import optuna
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from models.SARD_estimator import SARDModel, SARDLoss
from dataset_SARD import SARDData
from SARD_distill import fit_linear_model
from utils.utils import split_train_test, extract_best_model

# command line argument is which gpu to run on
DEVICE = 'cuda:' + sys.argv[1]
# DEVICE = 'cuda:0'


def objective(trial):
    task = 'gp_mortality_grid_search'
    results_directory = pathlib.Path.cwd().joinpath('SavedModels', f'{task}')
    if not results_directory.exists():
        results_directory.mkdir(parents=True)

    data_folder = pathlib.Path.cwd().joinpath('data', 'plp_output', 'sparse_matrix_no_age_5_years')
    data = np.load(data_folder.joinpath('data.npy'), allow_pickle=True)
    data = data.item()
    feature_matrix = data['feature_matrix']
    dataset_dict = data['dataset_dict']

    batch_size = 64
    update_every = 512 // batch_size  # accumulate gradients
    # create pytorch datasets and dataloaders
    regression_results = results_directory.joinpath('linear_predictions.npy')
    if not regression_results.exists():
        indices = split_train_test(data['outcomes'])
        linear_predictions, _ = fit_linear_model(feature_matrix.coalesce(), data['good_feature_names'],
                                                 data['covariates'], data['outcomes'],
                                                 window_lengths=(30, 90, 180, 365, 1095))
        np.save(results_directory.joinpath('linear_predictions.npy'), linear_predictions)
        np.save(results_directory.joinpath('indices.npy'), indices)
    else:
        linear_predictions = np.load(results_directory.joinpath('linear_predictions.npy'))
        indices = np.load(results_directory.joinpath('indices.npy'), allow_pickle=True).item()

    train_dataset = SARDData(indices, data['outcomes'].values, linear_predictions, stage='train')
    val_dataset = SARDData(indices, data['outcomes'].values, linear_predictions, stage='validation')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=10, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=10, pin_memory=True)

    fit_params = {'lr': 0.01,
                  'epochs': 5}
    output_dir = results_directory.joinpath('distill')
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    fit_params['results_dir'] = output_dir

    loss_distill = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.FloatTensor([
            len(dataset_dict['outcomes_filt']) / dataset_dict['outcomes_filt'].sum() - 1]), reduction='mean').to(DEVICE)
    loss_classifier = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.FloatTensor([len(dataset_dict['outcomes_filt']) / dataset_dict['outcomes_filt'].sum() - 1]),
        reduction='mean').to(DEVICE)

    embedding_dim = trial.suggest_int('embedding_dim', 32, 196, 32)
    n_heads = trial.suggest_int('n_heads', 2, 6, 2)
    attn_depth = trial.suggest_int('attn_depth', 2, 6, 1)
    dropout = trial.suggest_discrete_uniform('dropout', 0, 0.3, 0.05)
    alpha = trial.suggest_discrete_uniform('alpha', 0, 0.3, 0.05)
    # embedding_dim = 32
    # n_heads = 2
    # attn_depth = 3
    # dropout = 0.15
    # alpha = 0.25

    SARD_distill = SARDLoss(loss_distill, loss_classifier, alpha, distill=True)
    SARD_final = SARDLoss(loss_distill, loss_classifier, alpha)
    model_parameters = {'embedding_dim': embedding_dim, 'n_heads': n_heads, 'attn_depth': attn_depth,
                        'dropout': dropout, 'alpha': alpha}
    folder_name = '_'.join([f'{key}:{val}' for key, val in model_parameters.items()])

    results_dir = output_dir.joinpath(folder_name)
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    fit_params['results_dir'] = results_dir
    fit_params['prefix'] = 'grid_search_distill'
    fit_params['update_every'] = update_every
    model_parameters['use_mask'] = True
    model_parameters['concept_embedding_path'] = None
    model_parameters['n_features'] = feature_matrix.shape[1]
    model = SARDModel(fit_parameters=fit_params, model_parameters=model_parameters, data_dictionary=dataset_dict,
                      device=DEVICE, criterion=SARD_distill, linear_predictions=linear_predictions)
    model.fit(train_dataloader, val_dataloader)

    del model

    best_model_file = extract_best_model(fit_params['results_dir'])

    best_model_dict = torch.load(best_model_file)
    best_state_dict = best_model_dict['model_state_dict']
    fit_params['prefix'] = 'grid_search_finetune'
    fit_params['update_every'] = update_every

    finetune_dir = results_dir.joinpath('finetune')
    fit_params['results_dir'] = finetune_dir
    if not finetune_dir.exists():
        finetune_dir.mkdir(parents=True)
    fit_params['learning_rate'] = 3e-5
    model_finetune = SARDModel(fit_parameters=fit_params, model_parameters=model_parameters,
                               data_dictionary=dataset_dict, linear_predictions=linear_predictions,
                               device=DEVICE, criterion=SARD_final)
    model_finetune.load_state_dict(best_state_dict, strict=False) # not strict because the loss is different than in the distillation
    model_finetune.optimizer = torch.optim.Adam(model_finetune.model.parameters(), lr=3e-5)  # think this is overwritten by loading state dict

    model_finetune.fit(train_dataloader, val_dataloader)

    best_finetuned = extract_best_model(finetune_dir)
    state_dict = torch.load(best_finetuned)['model_state_dict']
    model_finetune.load_state_dict(state_dict)

    predictions, outcomes = model_finetune.predict_proba(val_dataloader)

    auc = roc_auc_score(outcomes, predictions)

    return auc


if __name__ == '__main__':
    search_space = {'embedding_dim': [32, 64, 96, 128, 160],
                    'n_heads': [2, 3, 4, 5, 6],
                    'attn_depth': [2, 3, 4, 5, 6],
                    'dropout': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
                    }
    study = optuna.create_study(storage='sqlite:///trial_run.db', study_name='SARD_grid_search', load_if_exists=True,
                                direction='maximize', sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(objective,  n_trials=100)
