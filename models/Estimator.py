import pathlib
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Batch
from tqdm import tqdm

from utils.utils import extract_best_model


class Estimator:
    """
    A class that wraps around pytorch models. Using this class I can quickly add pytorch models without
    having to write much code.
    """

    def __init__(self, model, model_parameters, fit_parameters,
                 optimizer=torch.optim.AdamW, scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                 criterion=nn.BCEWithLogitsLoss(), device='cpu'):
        """

        Parameters
        ----------
        model : nn.Module                   A pytorch model with a forward or __call__ method
        model_parameters : dict             The parameters to pass on to the pytorch model
        fit_parameters : dict               The parameters for the estimator
        optimizer :                         A pytorch optimizer, defaults to AdamW
        scheduler :                         A pytorch learning rate scheduler, default is reduce on plateau
        criterion :                         A pytorch loss function, default is BCEWithLogitsLoss
        device :                            Device to use, either 'cpu' or 'cuda:x' where x is number of gpu
        """
        self.model = model(**model_parameters)
        self.model_parameters = model_parameters

        self.epochs = fit_parameters.get('epochs', 5)
        self.learning_rate = fit_parameters.get('lr', 2e-4)
        self.weight_decay = fit_parameters.get('weight_decay', 1e-5)
        self.results_dir = pathlib.Path(fit_parameters.get('results_dir', './results'))
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True)

        self.prefix = fit_parameters.get('prefix', 'Model')
        self.previous_epochs = fit_parameters.get('previous_epochs', 0)
        self.clip = fit_parameters.get('clip', 5)

        # exclude certain parameter types from weight decay. Biases, norm parameters and embedding weights.
        def needs_wd(name):
            return all(x not in name for x in ['embedding', 'embedder', 'tokenizer', '.norm', '.bias'])

        parameters_with_wd = [v for k, v in self.model.named_parameters() if needs_wd(k)]
        parameters_without_wd = [v for k, v in self.model.named_parameters() if not needs_wd(k)]
        self.optimizer = optimizer(params=([{'params': parameters_with_wd},
                                            {'params': parameters_without_wd,
                                             'weight_decay': 0.0}]), lr=self.learning_rate,
                                   weight_decay=self.weight_decay)

        self.scheduler = scheduler(self.optimizer, mode='min', factor=0.1, patience=1)

        self.criterion = criterion
        self.device = device
        self.early_stopping = EarlyStopping(patience=3, verbose=True, path=f'')
        self.model.to(device)

    def fit(self, dataloader, test_dataloader):
        """
        Function that fit's a model to data loaded with a pytorch dataloader. It uses early stopping with data
        loaded with test_dataloader

        Parameters
        ----------
        dataloader :        A pytorch dataloader
        test_dataloader :   A pytorch dataloader used for early stopping, typically loading the validation set

        Returns
        -------
        self :              Returns itself so I can chain together operations like fit().score()

        """
        val_losses = []
        val_AUCs = []
        all_lr = []
        for epoch in range(self.epochs):
            self.fit_epoch(dataloader)
            val_loss, val_auc = self.score(test_dataloader)

            current_epoch = epoch + 1 + self.previous_epochs
            lr = self.optimizer.param_groups[0]["lr"]
            print(f'Epochs: {current_epoch} | Val AUC: {val_auc:.3f} | Val loss: {val_loss:.3f} | '
                  f'LR: {lr}')

            all_lr.append(lr)
            if self.scheduler:
                self.scheduler.step(val_loss)
            val_losses.append(val_loss)
            val_AUCs.append(val_auc)
            self.early_stopping(val_loss)
            if self.early_stopping.improved:
                torch.save({'model_state_dict': self.model.state_dict(),
                            'model_hyperparameters': self.model_parameters,
                            'epoch': current_epoch,
                            'learning_rates': all_lr},
                           self.results_dir.joinpath(f'{self.prefix}_epochs:{current_epoch}_auc:{val_auc:.4f}_'
                                                     f'val_loss:{val_loss:.4f}'))
            if self.early_stopping.early_stop:
                print('Early stopping, validation loss stopped improving')
                np.savetxt(self.results_dir.joinpath('log.txt'), np.array([val_losses, val_AUCs]))
                self.load_best_weights()
                return self
        np.savetxt(self.results_dir.joinpath('log.txt'), np.array([val_losses, val_AUCs]))
        self.load_best_weights()
        return self

    def fit_epoch(self, dataloader):
        """
        Fit's one epoch. An epoch is one round through the data you have available.

        Parameters
        ----------
        dataloader :        A pytorch dataloader

        Returns
        -------

        """
        t, batch_loss = time.time(), 0
        self.model.train()
        for batch_num, (batch, target) in enumerate(tqdm(dataloader)):
            batch = self._batch_to_device(batch)
            target = self._batch_to_device(target)
            y_pred = self.model(batch)
            loss = self.criterion(y_pred, target)
            batch_loss += loss.item()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
        print(f'Training Loss: {batch_loss / batch_num:.3f} | Time: {time.time() - t:.2f}')

    def score(self, dataloader):
        """
        Scores the model after fitting

        Parameters
        ----------
        dataloader :    A pytorch dataloader, typically validation or test data

        Returns
        -------
        validation loss: The loss from the supplied loss function
        auc:             The aread under the receiver operating characteristics curve.

        """
        self.model.eval()
        with torch.no_grad():
            validation_loss = []
            predictions = []
            targets = []
            for batch_num, (batch, target) in enumerate(tqdm(dataloader)):
                batch = self._batch_to_device(batch)
                target = self._batch_to_device(target)
                y_pred = self.model(batch)
                if isinstance(y_pred, list):
                    predictions += y_pred[0].tolist()
                else:
                    predictions += y_pred.tolist()
                if isinstance(target, list):
                    targets += target[0].tolist()
                else:
                    targets += target.tolist()
                validation_loss.append(self.criterion(y_pred, target).item())
            validation_loss = np.mean(validation_loss)
        self.model.train()
        return validation_loss, roc_auc_score(targets, predictions)

    def predict_proba(self, dataloader):
        """
        Gives predictions on data loaded with a dataloader. Predictions are between 0 and 1

        Parameters
        ----------
        dataloader :        A pytorch dataloader

        Returns
        -------
        predictions :       A torch tensor with predictions per sample

        """
        self.model.eval()
        with torch.no_grad():
            predictions = []
            for batch_num, (batch, _) in enumerate(tqdm(dataloader)):
                batch = self._batch_to_device(batch)
                y_pred = self.model(batch)
                if isinstance(y_pred, list):
                    y_pred = y_pred[0]
                predictions += torch.sigmoid(y_pred).tolist()
        self.model.train()
        return predictions

    def predict(self, dataloader):
        """
        Predicts a class label on data loaded with dataloader.

        #TODO need to make sure this works.
        #TODO Why am I taking an argmax if the dimension of predictions is num_samples x 1 ?

        Parameters
        ----------
        dataloader : A pytorch datalaoder

        Returns
        -------
        predicted_class : A pytorch tensors with predicted class label.

        """
        predictions = self.predict_proba(dataloader)
        predicted_class = np.argmax(predictions)
        return predicted_class

    def _batch_to_device(self, batch):
        """
        Sends data in batch to device. If batch is a list it goes recursively through each element in it's list and
        sends it to the device.

        Parameters
        ----------
        batch :         The batch data. Can be a tensor, a list or a pytorch geometric Batch

        Returns
        -------

        """
        if isinstance(batch, torch.Tensor) or isinstance(batch, Batch):
            batch = batch.to(self.device)
        else:
            for ix, b in enumerate(batch):
                if isinstance(b, torch.Tensor):
                    b = b.to(self.device)
                elif isinstance(b, list):
                    b = self._batch_to_device(b)
                else:
                    Warning('Unsupported type found in batch')
                batch[ix] = b
        return batch

    def fit_whole_training_set(self, dataloader, epochs, learning_rates):
        """
        Fits the model on the whole training set for a certain number of epochs. This is done after finding the
        best set of hyperparameters. Here it's not possible to use early stopping so we fit for the same number
        of epochs as the best epoch from the hyperparameter search. We use the same lr schedule as
        was used during hyperparameter search.

        Parameters
        ----------
        dataloader :            A pytorch dataloader
        epochs :                Number of epochs to fit
        learning_rates :        Optional, learning rates per epoch to match lr schedule of hyperparameter search

        Returns
        -------

        """
        for epoch in range(epochs):
            if learning_rates:
                self.optimizer = self.adjust_learning_rate(self.optimizer, learning_rates, epoch)
                print(f'Learning rate for epoch {epoch + 1} is {learning_rates[epoch]}')
            self.fit_epoch(dataloader)
            torch.save({'model_state_dict': self.model.state_dict(),
                        'model_hyperparameters': self.model_parameters,
                        'epoch': epoch},
                       self.results_dir.joinpath(f'{self.prefix}_epochs:{epoch}_whole_training_set'))

    @staticmethod
    def adjust_learning_rate(optimizer, learning_rates, epoch):
        """
        Changes the learning rate manually per epochs. Used when refitting on whole dataset.

        Parameters
        ----------
        optimizer :                 A pytorch optimizer
        learning_rates : list       A list of learning rates
        epoch : int                 Which epoch we change the learning rate for

        Returns
        -------

        """
        new_learning_rate = learning_rates[epoch]

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_learning_rate
        return optimizer

    def load_best_weights(self, directory=None, epoch=None):
        """
        Searches through all checkpoints in directory and loads the weights from the checkpoint with the lowest
        validation loss.

        Parameters
        ----------
        directory : Optional. A pathlib directory to search. Otherwise it uses self.results_dir
        epoch :     Optional. Force to load weights from a certain epoch.

        Returns
        -------

        """
        if directory:
            results_dir = directory
        else:
            results_dir = self.results_dir
        if epoch:
            best_model_file = self.extract_model_from_epoch(results_dir, epoch)
        else:
            best_model_file = extract_best_model(results_dir, metric='val_loss')
        best_model = torch.load(best_model_file)
        state_dict = best_model['model_state_dict']
        epoch = best_model['epoch']
        self.model.load_state_dict(state_dict)
        print(f'Loaded best model from epoch: {epoch}')

    @staticmethod
    def extract_model_from_epoch(directory, epoch):
        """
        Get's a checkpoint path from a certain epoch.

        Parameters
        ----------
        directory : A pathlib directory
        epoch :     Epoch to load weights from

        Returns
        -------
        A pathlib path to checkpoint from the chosen epoch.

        """
        fnames = []
        for f in directory.rglob(f'*epochs:{epoch}_whole_training_set*'):
            fnames.append(f)
        return fnames[0]


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
        self.improved = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = pathlib.Path(path)
        self.trace_func = trace_func
        self.previous_score = 0

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.improved = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.improved = False
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                if (score - self.previous_score) > 0:  # if loss is still decreasing don't stop
                    self.early_stop = False
                else:
                    self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.improved = True
        self.previous_score = score
