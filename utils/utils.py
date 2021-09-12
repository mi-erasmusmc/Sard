import json

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

sns.set_theme()

def extract_best_model(directory, metric='val_loss'):
    """
    Extract best model from a directory with checkpoints

    Parameters
    ----------
    directory :        Pathlib Path, directory where model checkpoints have been stored
    metric :           Which metric to use, either 'auc' or 'val_loss'

    Returns
    -------
    best_model_file :   Pathlib path to file of best model

    """
    if metric is None:
        metric = 'auc'
        direction = 'max'
    elif metric == 'auc':
        direction = 'max'
    elif metric == 'val_loss':
        direction = 'min'
    else:
        ValueError(f'Unknown metric supplied. Needs to be either "auc" or "val_loss" but {metric} was given')
    metric_value, fnames = [], []
    for f in directory.rglob('*' + metric + '*'):
        l = f.name.split(metric + ':')
        metric_value.append(float(l[1].split('_')[0]))
        fnames.append(f)
    if direction == 'max':
        best_index = np.argmax(metric_value)
    elif direction == 'min':
        best_index = np.argmin(metric_value)
    best_model_file = fnames[best_index]
    return best_model_file


def create_sequence_data(covariates):
    """
    Takes in a covariate dataframe and creates sequences which are lists of patients of visits (lists) of concepts (lists)
    :param covariates :      dataframe covariate dataframe from plp package
    :return: sequences_list :     a nested list of patients of visits of concepts, concepts are integers
             visit_list :    list of patient visits with timeId of the visit
    """
    sequence_list = list(
        covariates.groupby(['rowIdPython', 'timeId'])['covariateId'].agg(list).groupby(level=0).agg(list))
    visit_list = list(covariates.groupby(['rowIdPython'])['timeId'].agg(lambda x: sorted(list(x.unique()))).values)
    return sequence_list, visit_list


def load_data(data_folder, name='plp_output'):
    """
    Loads data saved from plp

    Parameters
    ----------
    data_folder : pathlib Path     Folder where PLP output was saved
    name: str                      Name of data object

    Returns
    -------
    outcomes :                  Pandas Series with 1.0 for patients with outcome and 0.0 elsewhere
    feature_matrix_3d :         sparse matrix in Pytorch COO format. num patients X num features X num timepoints
    covariates : Dataframe      Covariates dataframe from PLP package
    good_feature_names :        covariate names
    dataset_dict :              dictionary with data in correct format for the deep model
    """
    # load output from plp data export
    plp_data = torch.load(data_folder.joinpath(name))
    population = plp_data['population']
    plp_data['outcomes'] = population.outcomeCount.astype(np.float32)
    plp_data['data'] = plp_data['data'].coalesce()
    old_covariate_ids = plp_data['map'].oldCovariateId
    covariate_ref = plp_data['covariateRef']
    feature_names = covariate_ref[covariate_ref.covariateId.isin(old_covariate_ids)].covariateName.values
    plp_data['feature_names'] = feature_names

    return plp_data


def plot_roc_curve(y_true, predictions, title='Dementia'):
    """
    Plots the ROC curve of many models together

    Parameters
    ----------
    y_true :            True labels
    predictions :       Dictionary with one (key, value) par for each model's predictions.

    Returns
    -------

    """

    plt.figure(figsize=(8, 6))
    for key, value in predictions.items():
        fpr, tpr, _ = roc_curve(y_true, value)
        auc = roc_auc_score(y_true, value)
        plt.plot(fpr, tpr, label=f'{key} AUC: {auc:.3f}')

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel('False positive rate', fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel('True positive rate', fontsize=15)

    plt.title(f'ROC Curve {title}', fontweight='bold', fontsize=15)

    plt.legend(prop={'size': 13}, loc='lower right')

    plt.show()


def plot_pr(y_true, predictions, title='dementia'):
    """
    Plots the Precision-recall curves for many models

    Parameters
    ----------
    y_true :        Ground truth from test set
    predictions :   Dictionary with one (key, value) par for each model's predictions.
    title : str     Title of plot

    Returns
    -------
    Plots the plot

    """

    plt.figure(figsize=(8, 6))
    for key, value in predictions.items():
        precision, recall, _ = precision_recall_curve(y_true, value)
        auprc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{key} AUPRC: {auprc:.3f}')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel('Recall', fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel('Precision', fontsize=15)

    plt.title(f'Precision-recall curve {title}', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='upper right')

class NpEncoder(json.JSONEncoder):
    """
    Class I use to change numpy datatypes to python datatypes before saving json
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)