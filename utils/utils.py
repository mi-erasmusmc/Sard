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


def plot_roc_curve(y_true, predictions):
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

    plt.title('ROC Curve', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')

    plt.show()


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

    plt.show()


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






