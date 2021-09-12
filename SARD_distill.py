import pathlib

import torch
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import models.RegressionGen as lr_models
from torch_lr_finder import LRFinder
from torch.utils.data import DataLoader

from utils.data_utils import window_data_sorted
from models.SARD_estimator import SARDModel, SARDLoss
from dataset_SARD import SARDData
from utils.utils import plot_roc_curve, extract_best_model, split_train_test


def fit_linear_model(feature_matrix, feature_names, indices, outcomes, window_lengths=(30, 365)):
    """
    Fits a windowed lasso regression model to the data. Code adapted from github.com/clinicalml/omop-learn

    Parameters
    ----------
    feature_matrix :            n_subjects X n_features X n_times sparse COO pytorch tensor
    feature_names : list        names of features
    indices :                   indices with train test split
    outcomes : dataframe        outcomes
    window_lengths : tuple      windows to use in windowed regression model

    Returns
    -------
    predictions :               predictions on whole dataset to use for reverse distillation
    model_info : dict           dictionary with model coefficients, 10 most important features and python object

    """

    # window the data using the window lengths specified below
    feature_matrix_counts, windowed_feature_names = window_data_sorted(
        window_lengths=list(window_lengths),
        feature_matrix=feature_matrix,
        all_feature_names=feature_names)

    feature_matrix_counts = feature_matrix_counts.T

    X_train = feature_matrix_counts[indices['train'] + indices['val']]
    X_test = feature_matrix_counts[indices['test']]
    y_train = outcomes[indices['train'] + indices['val']]
    y_test = outcomes[indices['test']]

    # train the regression model over several choices of regularization parameter
    param_grid = {'lr__C': np.logspace(-4, 4, 20)}
    grid_search = GridSearchCV(lr_models.gen_lr_pipeline(), param_grid, n_jobs=20, cv=5, scoring='roc_auc',
                               refit=True)
    grid_search.fit(X_train, y_train)
    print(f'Best validation AUC: {grid_search.best_score_:.3f}')

    clf_lr = grid_search.best_estimator_
    predictions = clf_lr.predict_proba(feature_matrix_counts)[:, 1]

    # pick the model with the best regularization, as measured by validation performance
    pred_lr = clf_lr.predict_proba(X_test)[:, 1]
    print('Linear Model Test AUC: {0:.3f}'.format(roc_auc_score(y_test, pred_lr)))

    coefs = clf_lr['lr'].coef_[0]

    highest_indices = coefs.argsort()[-10:][::-1]
    highest_features = [windowed_feature_names[h] for h in highest_indices]
    print(highest_features)
    model_info = {'coefficients': coefs, 'classifier': clf_lr, 'features': windowed_feature_names}

    return predictions, model_info


def distill_deep_model(feature_matrix, indices, dataset_dict, linear_predictions, outcomes, prefix='test_model',
                       device='cuda:0', find_lr=False, results_dir=None, load_from_checkpoint=False):
    """
    Fit deep model to match predictions from another model


    Parameters
    ----------
    feature_matrix :        3d sparse COO matrix, n_patients X n_features X n_times
    indices :               indices for train-val-test split
    dataset_dict :          dictionary with data that goes to deep model
    linear_predictions :    predictions from linear model
    outcomes :              outcomes
    prefix :                prefix to save files with
    device :                device to run model on
    find_lr :               use lr_finder to find optimal learning rate
    results_dir :           where to store results
    load_from_checkpoint :  if model should be started from a previous checkpoint

    Returns
    -------


    """
    batch_size = 512
    update_every = 512//batch_size  # accumulate gradients if batch size is smaller than 512

    # create pytorch datasets and dataloaders
    train_dataset = SARDData(indices, outcomes, linear_predictions, stage='train', distill=True)
    val_dataset = SARDData(indices, outcomes, linear_predictions, stage='validation', distill=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True,
                                  shuffle=True, num_workers=10)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, pin_memory=True,
                                shuffle=False, num_workers=0)

    # set some parameters for SARD,parameters below found with randomized search
    n_heads = 2
    embedding_dim = 32
    assert embedding_dim % n_heads == 0
    model_params = {
        'embedding_dim': embedding_dim,  # Dimension per head of visit embeddings
        'n_heads': n_heads,  # Number of self-attention heads
        'attn_depth': 2,  # Number of stacked self-attention layers
        'dropout': 0.0,  # Dropout rate for both self-attention and the final prediction layer
        'use_mask': True,  # Only allow visits to attend to other actual visits, not to padding visits
        'concept_embedding_path': None,  # if unspecified, uses default Torch embeddings
        'n_features': feature_matrix.shape[1]
    }

    # Set up fixed model parameters, loss functions, and build the model on the GPU
    fit_params = {'lr': 0.01,  # lr found with lr_finder
                  'epochs': 1,
                  'prefix': prefix,
                  'update_every': update_every}
    output_dir = results_dir.joinpath('distill')
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    fit_params['results_dir'] = output_dir

    pos_weight = torch.FloatTensor([(outcomes[indices['train']].shape[0] - outcomes[indices['train']].sum()) / outcomes[indices['train']].sum()])
    loss_distill = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')

    SARD_distill = SARDLoss(loss_distill, distill=True)

    model = SARDModel(fit_parameters=fit_params, model_parameters=model_params, data_dictionary=dataset_dict,
                      device=device, criterion=SARD_distill, linear_predictions=linear_predictions
                      )
    model.to(device)
    if find_lr:
        lr_finder = LRFinder(model.model, model.optimizer, model.criterion, device=device)
        lr_finder.range_test(train_dataloader, start_lr=1e-6, end_lr=1, num_iter=100)
        lr_finder.plot(log_lr=False)
        lr_finder.reset()
        return

    if load_from_checkpoint:
        epoch = []
        fnames = []
        for f in output_dir.rglob('*epochs*'):
            epoch.append(int(f.name.split(':')[1].split('_')[0]))
            fnames.append(f)
        latest_epoch_index = np.argmax(epoch)
        latest_epoch_file = fnames[latest_epoch_index]
        latest_epoch = epoch[latest_epoch_index]
        model.previous_epochs = latest_epoch
        state_dict = torch.load(latest_epoch_file)['model_state_dict']
        model.load_state_dict(state_dict)
    model.fit(train_dataloader, val_dataloader)


def fine_tune_deep_model(indices, dataset_dict, linear_predictions, model_dir=pathlib.Path('./SavedModels'),
                         device='cuda:0', prefix=''):
    """

    Fine tune deep model that has already been fit to linear model predictions

    Parameters
    ----------
    indices :               indices with train-test split
    dataset_dict :          data for deep model
    linear_predictions :    predictions from linear model
    model_dir :             dir where distilled model was saved
    device :                which gpu to use

    Returns
    -------

    """
    # load best model from earlier
    distill_directory = pathlib.Path(model_dir).joinpath('distill')
    best_model_file = extract_best_model(distill_directory)

    best_model_dict = torch.load(best_model_file)
    best_state_dict = best_model_dict['model_state_dict']
    hyperparameters = best_model_dict['model_hyperparameters']

    # training hyperparameter
    batch_size = 512
    # create pytorch datasets and dataloaders
    train_dataset = SARDData(indices, outcomes, linear_predictions, stage='train')
    val_dataset = SARDData(indices, outcomes, linear_predictions, stage='validation')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=10)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0)

    # pos weight is num_negative_samples / num_positive_samples
    pos_weight = torch.FloatTensor([(outcomes[indices['train']].shape[0] - outcomes[indices['train']].sum()) / outcomes[indices['train']].sum()])
    loss_distill = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean').to(device)
    loss_classifier = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean').to(device)
    best_state_dict['criterion.classifier_loss.pos_weight'] = pos_weight

    fit_params = {'lr': 3e-05,  # found with LR finder
                  'epochs': 1,
                  'prefix': prefix,
                  'update_every': 512//batch_size}

    # alpha_values = [0.0, 0.05, 0.1, 0.15, 0.2]
    alpha_values = [0.0]
    for alpha in alpha_values:
        SARD_loss = SARDLoss(loss_distill, loss_classifier, alpha=alpha, distill=False)
        fit_params['results_dir'] = model_dir.joinpath('fine-tune', f'alpha:{alpha}')

        model = SARDModel(model_parameters=hyperparameters, fit_parameters=fit_params, data_dictionary=dataset_dict,
                          device=device, criterion=SARD_loss)
        model.load_state_dict(best_state_dict)

        if not fit_params['results_dir'].exists():
            fit_params['results_dir'].mkdir(parents=True)

        model.fit(train_dataloader, val_dataloader)


def load_data(data_folder):
    """

    Parameters
    ----------
    data_folder : pathlib Path     Folder where PLP output was saved

    Returns
    -------
    outcomes :                  Pandas Series with 1.0 for patients with outcome and 0.0 elsewhere
    feature_matrix_3d :         sparse matrix in Pytorch COO format. num patients X num features X num timepoints
    covariates : Dataframe      Covariates dataframe from PLP package
    good_feature_names :        covariate names
    dataset_dict :              dictionary with data in correct format for the deep model
    """
    # load output from plp data export
    plp_output = torch.load(data_folder.joinpath('plp_output'))
    population = plp_output['population']
    outcomes = population.outcomeCount.astype(np.float32)
    feature_matrix_3d = plp_output['data'].coalesce()
    feature_map = plp_output['map']
    covariate_ref = plp_output['covariateRef']
    covariates = plp_output['covariates']

    # process data for deep model, slightly adapted code from github.com/clinicalml/omop-learn
    person_ixs, code_ixs, time_ixs = feature_matrix_3d.indices()
    all_codes_tensor = code_ixs
    people = sorted(np.unique(person_ixs))
    person_indices = np.searchsorted(person_ixs, people)
    person_indices = np.append(person_indices, len(person_ixs))  # adds end to array
    person_chunks = [
        time_ixs[person_indices[i]: person_indices[i + 1]]
        for i in range(len(person_indices) - 1)
    ]
    visit_chunks = []
    visit_times_raw = []
    for i, chunk in enumerate(person_chunks):
        visits = sorted(np.unique(chunk))
        visit_indices_local = np.searchsorted(sorted(chunk), visits)
        visit_indices_local = np.append(
            visit_indices_local,
            len(chunk)
        )
        visit_chunks.append(visit_indices_local)
        visit_times_raw.append(visits)
    n_visits = {i: len(j) for i, j in enumerate(visit_times_raw)}
    old_covariate_ids = feature_map.oldCovariateId
    good_feature_names = covariate_ref[covariate_ref.covariateId.isin(old_covariate_ids)].covariateName.values
    visit_days_rel = visit_times_raw
    dataset_dict = {
        'all_codes_tensor': all_codes_tensor,  # A tensor of all codes occurring in the dataset
        'person_indices': person_indices,
        # A list of indices such that all_codes_tensor[person_indices[i]: person_indices[i+1]] are the codes assigned to the ith patient
        'visit_chunks': visit_chunks,
        # A list of indices such that all_codes_tensor[person_indices[i]+visit_chunks[j]:person_indices[i]+visit_chunks[j+1]] are the codes assigned to the ith patient during their jth visit
        'visit_time_rel': visit_days_rel,  # A list of times (as measured in days to the prediction date) for each visit
        'n_visits': n_visits,  # A dict defined such that n_visits[i] is the number of visits made by the ith patient
        'outcomes_filt': outcomes,
        # A pandas Series defined such that outcomes_filt.iloc[i] is the outcome of the ith patient
    }

    return outcomes, feature_matrix_3d, covariates, good_feature_names, dataset_dict


if __name__ == '__main__':
    run_linear = True
    device = 'cuda:0'
    distill = True
    task = 'gp_mortality'
    prefix = 'test'
    data_from_cache = False
    data_folder = pathlib.Path.cwd().joinpath('data', 'plp_output')


    results_directory = pathlib.Path.cwd().joinpath('SavedModels', f'{task}')
    if not results_directory.exists():
        results_directory.mkdir(parents=True)

    if not data_from_cache:
        outcomes, feature_matrix_3d, covariates, good_feature_names, dataset_dict = load_data(data_folder)
        data = {'outcomes': outcomes, 'feature_matrix': feature_matrix_3d, 'covariates': covariates,
                'good_feature_names': good_feature_names, 'dataset_dict': dataset_dict}
        np.save(data_folder.joinpath('data'), data)
    else:
        data = np.load(data_folder.joinpath('data.npy'), allow_pickle=True)
        data = data.item()
        outcomes, feature_matrix_3d = data['outcomes'],  data['feature_matrix']
        covariates, good_feature_names = data['covariates'], data['good_feature_names']
        dataset_dict = data['dataset_dict']

    indices = split_train_test(outcomes)

    # fit linear model
    if run_linear:
        linear_predictions, linear_model = fit_linear_model(feature_matrix_3d.coalesce(), good_feature_names, indices,
                                                            outcomes, window_lengths=(30, 90, 180, 365, 1095))
        np.save(results_directory.joinpath('linear_predictions.npy'), linear_predictions)

    else:
        linear_predictions = np.load(results_directory.joinpath('linear_predictions.npy'))

    if distill:
        # fit deep learning model to linear model
        distill_deep_model(feature_matrix=feature_matrix_3d, indices=indices, dataset_dict=dataset_dict,
                           linear_predictions=linear_predictions, outcomes=outcomes, find_lr=False, device=device,
                           prefix=prefix, results_dir=results_directory, load_from_checkpoint=False)

    # fine tune already fitted deep model
    fine_tune_deep_model(indices=indices, dataset_dict=dataset_dict,
                         linear_predictions=linear_predictions, device=device, model_dir=results_directory)

    # find best finetuned model and test on test set
    finetuned_directory = results_directory.joinpath('fine-tune')

    # use AUC as metric to find best model here since magnitudes of losses with different alphas are not comparable
    best_model_file = extract_best_model(finetuned_directory, metric='auc')
    best_model = torch.load(best_model_file)
    hyperparameters = best_model['model_hyperparameters']
    state_dict = best_model['model_state_dict']
    hyperparameters['n_features'] = feature_matrix_3d.shape[1]

    pos_weight = torch.FloatTensor([len(dataset_dict['outcomes_filt']) / dataset_dict['outcomes_filt'].sum() - 1])
    loss_distill = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
        reduction='mean').to(device)
    loss_classifier = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
        reduction='mean').to(device)
    SARD_loss = SARDLoss(loss_distill, loss_classifier, alpha=0.15, distill=False)

    model = SARDModel(model_parameters=hyperparameters, data_dictionary=dataset_dict,
                      device=device, criterion=SARD_loss)
    model.load_state_dict(state_dict)

    test_dataset = SARDData(indices, outcomes, linear_predictions, stage='test')

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=512,
                                 shuffle=False, num_workers=10)

    predictions, y_true = model.predict_proba(test_dataloader)
    best_auc = roc_auc_score(y_true, predictions)
    prediction_dict = {'SARD': predictions, 'LASSO': linear_predictions[indices['test']]}
    plot_roc_curve(y_true, prediction_dict)
    np.savetxt(results_directory.joinpath('best_auc_SARD.txt'), np.array([best_auc]))
