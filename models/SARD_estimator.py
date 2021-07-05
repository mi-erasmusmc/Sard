import time

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from models.SARD_model import VTClassifer, VisitTransformer


class SARDModel(nn.Module):
    def __init__(self, model_parameters=None, data_dictionary=None, fit_parameters=None,
                 optimizer=torch.optim.Adam,
                 scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                 criterion=torch.nn.BCEWithLogitsLoss(),
                 linear_predictions=None,
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
        linear_predictions :    predictions of previously fit modelvalidation_batch_loss
        device :                which device to run model on
        """
        super(SARDModel, self).__init__()

        if fit_parameters is None:
            fit_parameters = {}
        self.is_fitted = False
        self.device = device
        self.data_dictionary = data_dictionary
        self.model_parameters = model_parameters

        self.base_model = VisitTransformer(device=device, **model_parameters)
        self.model = VTClassifer(self.base_model, device=device, **model_parameters)

        self.model.bert.set_data(torch.LongTensor(data_dictionary['all_codes_tensor']),
                                 data_dictionary['person_indices'], data_dictionary['visit_chunks'],
                                 data_dictionary['visit_time_rel'], data_dictionary['n_visits'])
        self.model.to(device)

        self.epochs = fit_parameters.get('epochs', 5)
        self.learning_rate = fit_parameters.get('lr', 2e-3)
        self.results_dir = fit_parameters.get('results_dir', './results')
        self.prefix = fit_parameters.get('prefix', 'test_SARD')
        self.update_every = fit_parameters.get('update_every', 1)  # to accumulate gradients
        self.previous_epochs = fit_parameters.get('previous_epochs', 0)  # if starting from previous checkpoint

        self.optimizer = optimizer(params=self.model.parameters(), lr=self.learning_rate)

        self.scheduler = scheduler(self.optimizer, mode='min', factor=0.1,
                                   patience=1)
        self.criterion = criterion
        if linear_predictions is not None:
            self.linear_predictions = torch.FloatTensor(linear_predictions)

    def fit(self, dataloader, test_dataloader):
        """
        Fit model

        """
        self.train()
        val_losses = []
        for epoch in range(self.epochs):
            t, batch_loss = time.time(), 0
            for batch_num, (batch, target) in enumerate(tqdm(dataloader)):
                p_range = batch.to(self.device)
                if len(target) > 1:
                    target = [t.to(self.device) for t in target]
                else:
                    target = target.to(self.device)

                y_pred = self.model(p_range)
                loss = self.criterion(y_pred, target)

                if batch_num % (10 * self.update_every) == 0:
                    print(f'Loss: {batch_loss / (10 * self.update_every):.3f} | Time: {time.time() - t:.2f}')
                    t, batch_loss = time.time(), 0

                batch_loss += loss.item()
                loss.backward()

                if batch_num % self.update_every == 0:  # accumulate gradients
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            self.eval()
            val_auc, validation_loss = self.score(test_dataloader)
            current_epoch = epoch + 1 + self.previous_epochs
            print(f'Epochs: {current_epoch} | Val AUC: {val_auc:.3f} | Val distill loss: {validation_loss:.3f} | '
                  f'LR: {self.optimizer.param_groups[0]["lr"]}')
            if self.scheduler:
                self.scheduler.step(validation_loss)
            val_losses.append(val_auc)
            torch.save({'model_state_dict': self.state_dict(),
                        'model_hyperparameters': self.model_parameters},
                       self.results_dir.joinpath(f'{self.prefix}_epochs:{current_epoch}_auc:{val_auc:.4f}_'
                                                 f'val_loss:{validation_loss:.4f}'))
            self.train()
        np.savetxt(self.results_dir.joinpath('log.txt'), val_losses)

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
                patients = batch.to(self.device)
                y_pred = self.model(patients)
                y_proba = torch.sigmoid(y_pred)
                all_patients.append(patients.cpu().numpy())
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
                p_range = batch.to(self.device)
                if len(target) > 1:
                    target = [t.to(self.device) for t in target]
                else:
                    target = target.to(self.device)
                y_pred = self.model(p_range)
                preds_test += y_pred.tolist()
                all_patients += p_range.tolist()
                validation_batch_loss.append(self.criterion(y_pred, target).item())
            validation_loss = np.mean(validation_batch_loss)
            return roc_auc_score(self.data_dictionary['outcomes_filt'][all_patients], preds_test), validation_loss


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


