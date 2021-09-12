import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm

from models.Estimator import EarlyStopping
from models.SARD_model import VTClassifer, VisitTransformer


class SARDModel(nn.Module):
    """
    An estimator class with fit and predict methods to use with original SARD implementation.
    """

    def __init__(self, model_parameters=None, data_dictionary=None, fit_parameters=None,
                 optimizer=torch.optim.Adam,
                 scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                 criterion=torch.nn.BCEWithLogitsLoss(),
                 device='cuda:0'):
        """

        Parameters
        ----------
        model_parameters :      dictionary with model parameters
        data_dictionary :       dictionary with data for deep model
        fit_parameters :        other fitting parameters
        optimizer :             which optimizer to use, default Adam
        scheduler :             which scheduler to use, default ReduceLROnPlateau
        criterion :             Loss to use
        device :                which device to run model on
        """
        super(SARDModel, self).__init__()

        if fit_parameters is None:
            fit_parameters = {}
        self.is_fitted = False
        self.device = device
        self.data_dictionary = data_dictionary
        self.model_parameters = model_parameters

        self.bert = VisitTransformer(device=device, **model_parameters)
        self.model = VTClassifer(self.bert, device=device, **model_parameters)

        self.model.bert.set_data(torch.LongTensor(data_dictionary['all_codes_tensor']),
                                 data_dictionary['person_indices'], data_dictionary['visit_chunks'],
                                 data_dictionary['visit_time_rel'], data_dictionary['n_visits'])
        self.model.to(device)

        self.epochs = fit_parameters.get('epochs', 5)
        self.learning_rate = fit_parameters.get('lr', 2e-3)
        self.results_dir = fit_parameters.get('results_dir', './results')
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True)
        self.prefix = fit_parameters.get('prefix', 'test_SARD')
        self.update_every = fit_parameters.get('update_every', 1)  # to accumulate gradients
        self.previous_epochs = fit_parameters.get('previous_epochs', 0)  # if starting from previous checkpoint

        self.optimizer = optimizer(params=self.model.parameters(), lr=self.learning_rate)

        self.scheduler = scheduler(self.optimizer, mode='min', factor=0.1,
                                   patience=1)
        self.criterion = criterion
        self.early_stopping = EarlyStopping(patience=3, verbose=True, path=f'')

    def fit(self, dataloader, test_dataloader):
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True)
        val_losses = []
        val_AUCs = []
        for epoch in range(self.epochs):
            self.fit_epoch(dataloader)
            val_auc, val_loss = self.score(test_dataloader)

            current_epoch = epoch + 1 + self.previous_epochs
            print(f'Epochs: {current_epoch} | Val AUC: {val_auc:.2f} | Val loss: {val_loss:.2f} | '
                  f'LR: {self.optimizer.param_groups[0]["lr"]}')
            if self.scheduler:
                self.scheduler.step(val_loss)
            val_losses.append(val_loss)
            val_AUCs.append(val_auc)
            self.early_stopping(val_loss)
            if self.early_stopping.improved:
                torch.save({'model_state_dict': self.model.state_dict(),
                            'model_hyperparameters': self.model_parameters,
                            'epoch': current_epoch},
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
        batch_loss = 0
        self.model.train()
        for batch_num, (batch, target) in enumerate(tqdm(dataloader)):
            batch = self._batch_to_device(batch)
            target = self._batch_to_device(target)
            y_pred = self.model(batch)
            loss = self.criterion(y_pred, target)
            batch_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _batch_to_device(self, batch):
        for ix, b in enumerate(batch):
            if isinstance(b, torch.Tensor):
                b = b.to(self.device)
            elif isinstance(b, list):
                b = self._batch_to_device(b)
            else:
                Warning('Unsupported type found in batch')
            batch[ix] = b
        return batch

    def predict_proba(self, dataloader):
        """
        Outputs the predicted risk probabilities

        """
        predictions = []
        all_patients = []
        outcomes = torch.FloatTensor(self.data_dictionary['outcomes_filt'].values)
        self.eval()
        with torch.no_grad():
            for batch_num, (batch, _) in enumerate(tqdm(dataloader)):
                batch = self._batch_to_device(batch)
                y_pred = self.model(batch)
                y_proba = torch.sigmoid(y_pred)
                all_patients.append(batch[0].cpu().numpy())
                predictions.append(y_proba.cpu().numpy())
        predictions = np.concatenate(predictions)
        all_patients = np.concatenate(all_patients)
        outcomes = outcomes[all_patients]
        return predictions, outcomes

    def predict(self, dataloader):
        """
        Outputs the predicted class
        """
        predictions = self.predict_proba(dataloader)
        predictions = int(predictions >= 0.5)
        return predictions

    def score(self, dataloader):
        """Score predictions, both AUC and loss"""
        with torch.no_grad():
            preds_test, all_patients = [], []
            validation_batch_loss = []
            for batch_num, (batch, target) in enumerate(dataloader):
                batch = self._batch_to_device(batch)
                target = self._batch_to_device(target)
                y_pred = self.model(batch)
                preds_test += y_pred.tolist()
                all_patients += batch[0].tolist()
                validation_batch_loss.append(self.criterion(y_pred, target).item())
            validation_loss = np.mean(validation_batch_loss)
            return roc_auc_score(self.data_dictionary['outcomes_filt'][all_patients], preds_test), validation_loss

    def load_best_weights(self, directory=None, strict=True):
        if directory:
            results_dir = directory
        else:
            results_dir = self.results_dir
        best_model_file = self.extract_best_model(results_dir, metric='val_loss')
        best_model = torch.load(best_model_file)
        state_dict = best_model['model_state_dict']
        epoch = best_model['epoch']
        self.model.load_state_dict(state_dict, strict=strict)
        print(f'Loaded best model from epoch: {epoch}')

    @staticmethod
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


class SARDLoss(nn.Module):
    def __init__(self, distill_loss, classifier_loss=None, alpha=0, distill=False):
        """

        Loss class for SARD

        Parameters
        ----------
        distill_loss :      Loss to use with previously fitted model predictions
        classifier_loss :   Loss to use with outcomes
        alpha :             How to blend the losses, if alpha is 0 the loss is only classifier_loss
        distill :           If we are distilling or not
        """
        super(SARDLoss, self).__init__()

        self.distill_loss = distill_loss
        self.classifier_loss = classifier_loss
        self.alpha = alpha
        self.distill = distill

    def forward(self, input, target):
        labels, linear_predictions = target
        if self.distill:
            return self.distill_loss(input, linear_predictions)
        else:
            return self.classifier_loss(input, labels) + self.alpha * self.distill_loss(input, linear_predictions)
