<<<<<<< HEAD
import json

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

sns.set_theme()


def extract_best_model(directory, metric='val_loss'):
    """
    Extracts best model from a directory, using a metric to define what is best
=======
import pathlib
import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = pathlib.Path(path)
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def extract_best_model(directory, metric='val_loss'):
    """
>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66

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


<<<<<<< HEAD
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
=======
def plot_roc_curve(y_true, predictions):
>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66
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

<<<<<<< HEAD
    plt.title(f'ROC Curve {title}', fontweight='bold', fontsize=15)
=======
    plt.title('ROC Curve', fontweight='bold', fontsize=15)
>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66
    plt.legend(prop={'size': 13}, loc='lower right')

    plt.show()


<<<<<<< HEAD
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
=======
def plot_calibration(y_true, deep_predictions, linear_predictions):

    deep_prop_true, deep_prop_pred = calibration_curve(y_true, deep_predictions, n_bins=10, normalize=False)
    linear_prop_true, linear_prop_pred = calibration_curve(y_true, linear_predictions, n_bins=10, normalize=False)

    fig = plt.figure(figsize=(8, 6))

    plt.plot(deep_prop_pred, deep_prop_true, label=f'SARD')
    plt.plot(linear_prop_pred, linear_prop_true, label=f'Linear model')
    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel('Predicted value', fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel('Fraction of positives', fontsize=15)

    plt.title('Calibration', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')
>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66

    plt.show()


<<<<<<< HEAD
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
=======
def split_train_test(outcomes, test_size=0.2, val_size=0.2, seed=42):
    """
    Get's a dictionary of indices for train-validation-test split
    Parameters
    ----------
    outcomes :      list of target outcomes
    test_size :     size of test set, default 20%
    val_size :      size of validation set, default 20%
    seed :          random seed to have reproducible results

    Returns
    -------
    indices : dic  A dictionary with train, validation and test indices
    """
    all_indices = range(len(outcomes))
    indices_train, indices_test = train_test_split(all_indices, test_size=test_size, stratify=outcomes, random_state=seed)
    indices_train, indices_val = train_test_split(indices_train, test_size=val_size/(1-test_size),
                                                  stratify=outcomes[indices_train], random_state=seed)
    indices = {'train': indices_train, 'val': indices_val, 'test': indices_test}
    return indices






>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66
