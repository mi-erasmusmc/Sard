import numpy as np
import torch

from models.Estimator import Estimator
from models.Transformer import Transformer


class Distiller:
    """
    A class to use for distillation.
    """

    def __init__(self, estimator=Estimator, model=Transformer,
                 model_parameters=None, fit_parameters=None, device='cuda:0',
                 alpha=0):
        """

        Parameters
        ----------
        estimator :                  An estimator class with fit and predict_proba methods
        model :                      A pytorch model
        model_parameters : dict      the parameters of the pytorch model
        fit_parameters : dict        the model agnostic parameters for the estimator
        device:                      The device to use, defaults to cuda:0
        alpha:                       Alpha to use when mixing distill and finetune loss, default is 0.
        """

        self.alpha = alpha

        self.model_parameters = model_parameters
        self.fit_parameters = fit_parameters
        self.estimator = estimator
        self.model_type = model
        self.device = device

    def distill(self, distill_loss, train_dataloader, test_dataloader, distill_learning_rate):
        """
        Function that distills

        Parameters
        ----------
        distill_loss :              A pytorch loss
        train_dataloader :          Dataloader to iterate over training set
        test_dataloader :           Dataloader to iterate over validation set
        distill_learning_rate :     Learning rate for distillation.

        Returns
        -------

        """
        self.fit_parameters['lr'] = distill_learning_rate
        self.fit_parameters['results_dir'] = self.fit_parameters['results_dir'].joinpath('distill')
        self.model = self.estimator(model=self.model_type, model_parameters=self.model_parameters,
                                    fit_parameters=self.fit_parameters,
                                    device=self.device, criterion=distill_loss)
        self.model.fit(train_dataloader, test_dataloader)

    def distill_whole_training_set(self, distill_loss, train_dataloader, epochs, learning_rates):
        """
        Refits a model to the whole training set. Trains for the same number of epochs and with learning rates
        used during hyperparameter search.

        Parameters
        ----------
        distill_loss :              A pytorch loss function
        train_dataloader :          Dataloader to iterate over the whole training set.
        epochs :                    Number epochs to train for.
        learning_rates :            A list of learning rates to use per epoch

        Returns
        -------

        """
        self.fit_parameters['results_dir'] = self.fit_parameters['results_dir'].joinpath('distill')
        self.model = self.estimator(model=self.model_type, model_parameters=self.model_parameters,
                                    fit_parameters=self.fit_parameters,
                                    device=self.device, criterion=distill_loss)
        self.model.fit_whole_training_set(train_dataloader, epochs, learning_rates)

    def finetune(self, finetune_loss, train_dataloader, test_dataloader, finetune_lr):
        """
        Finetunes a model that has been fit to predictions from a teacher model

        Parameters
        ----------
        finetune_loss :         A pytorch loss function
        train_dataloader :      A training dataloader to iterate over training set
        test_dataloader :       A test dataloader to iterate over validation set
        finetune_lr :           The learning rate to use for the finetuning

        Returns
        -------

        """
        self.fit_parameters['results_dir'] = self.fit_parameters['results_dir'].parent.joinpath('finetune')
        self.fit_parameters['lr'] = finetune_lr
        self.model = self.estimator(model=self.model_type, model_parameters=self.model_parameters,
                                    fit_parameters=self.fit_parameters,
                                    device=self.device, criterion=finetune_loss)
        self.model.load_best_weights(directory=self.fit_parameters['results_dir'].parent.joinpath('distill'))
        self.model.fit(train_dataloader, test_dataloader)

    def finetune_whole_training_set(self, finetune_loss, train_dataloader, epochs, distill_epoch,
                                    learning_rates):
        """
        Finetunes on the whole training set for the same number of epochs and learning rates as used during
        hyperparameter search

        Parameters
        ----------
        finetune_loss :         A pytorch loss function
        train_dataloader :      Train dataloader to iterate over the whole training set
        epochs :                Number of epochs to train for
        distill_epoch :         From what epoch to select distill model
        learning_rates :        List of learning rates to use

        Returns
        -------

        """
        self.fit_parameters['results_dir'] = self.fit_parameters['results_dir'].parent.joinpath('finetune')
        self.model = self.estimator(model=self.model_type, model_parameters=self.model_parameters,
                                    fit_parameters=self.fit_parameters,
                                    device=self.device, criterion=finetune_loss)
        self.model.load_best_weights(directory=self.fit_parameters['results_dir'].parent.joinpath('distill'),
                                     epoch=distill_epoch)
        self.model.fit_whole_training_set(train_dataloader, epochs, learning_rates)

    def predict_proba(self, dataloader):
        """
        Create predictions from an already fitted model.

        Parameters
        ----------
        dataloader :    Dataloader to iterate over data to make predictions for

        Returns
        -------

        """
        preds = self.model.predict_proba(dataloader)
        return preds


def load_data(data_folder, name='plp_output'):
    """
    Loads the data saved from the plp package.

    Parameters
    ----------
    data_folder : pathlib Path     Folder where PLP output was saved
    name: str                      Name of data object

    Returns
    -------
    Dictionary with the following fields:

    outcomes :                  Pandas Series with 1.0 for patients with outcome and 0.0 elsewhere
    data :                      Sparse matrix in Pytorch COO format. num patients X num features X num timepoints
    covariates : Dataframe      Covariates dataframe from PLP package
    feature_names :             Covariate names
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


def process_data_deep_model(feature_matrix_3d, outcomes):
    """Process data to use for original implementation of SARD"""
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

    return dataset_dict
