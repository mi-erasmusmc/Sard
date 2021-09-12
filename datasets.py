import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data, Batch
from tqdm.auto import tqdm

from utils.data_utils import window_data_sorted, add_age_gender


class GraphDataset(InMemoryDataset):
    """
    Dataset to use for graph neural networks.
    """

    def __init__(self, root='/data/home/efridgeirsson/projects/dementia/data/sequence_dementia'):
        super(GraphDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.labels = self.data.y

    @property
    def num_features(self):
        return len(self.data.x.unique())

    @property
    def raw_file_names(self):
        return ['python_data']

    @property
    def processed_file_names(self):
        return ['dementia.dataset']

    def download(self):
        pass

    def process(self):
        data = torch.load(self.raw_paths[0])
        old_covariate_ids = data['map'].oldCovariateId
        covariate_ref = data['covariateRef']
        feature_names = covariate_ref[covariate_ref.covariateId.isin(old_covariate_ids)].covariateName.values
        window_lengths = (30, 180, 365)
        feature_matrix_counts, windowed_feature_names = window_data_sorted(
            window_lengths=list(window_lengths),
            feature_matrix=data['data'].coalesce(),
            all_feature_names=feature_names)
        feature_matrix_counts = feature_matrix_counts.T
        feature_matrix_counts.data = np.clip(feature_matrix_counts.data, 0, 1)  # counts to binary
        feature_matrix_counts, windowed_feature_names = add_age_gender(feature_matrix_counts,
                                                                       data['nonTemporalData'],
                                                                       windowed_feature_names,
                                                                       age_normalized=False)
        train_index = data['population'][data['population']['index'] >= 0].index.values
        test_index = data['population'][data['population']['index'] < 0.0].index.values

        encounter_data = feature_matrix_counts[:, :-4]
        demographic_data = feature_matrix_counts[:, -4:].toarray()

        scaler = StandardScaler()
        demographic_data[train_index, :-1] = scaler.fit_transform(demographic_data[train_index, :-1])
        demographic_data[test_index, :-1] = scaler.transform(demographic_data[test_index, :-1])
        outcomes = torch.as_tensor(data['population'].outcomeCount.values, dtype=torch.float32)
        demographic_data = torch.as_tensor(demographic_data, dtype=torch.float32)

        patients = [p for p in range(encounter_data.shape[0])]
        data_list = self.process_patient(patients, demographic_data, encounter_data, outcomes)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def process_patient(patient_idxs, demographic_data=None, encounter_data=None, outcomes=None):
        data = []
        for patient_idx in tqdm(patient_idxs):
            patient_data = encounter_data[patient_idx, :].toarray()
            source_nodes = torch.as_tensor(patient_data.nonzero()[1], dtype=torch.long)
            num_nodes = len(source_nodes)
            source_nodes = source_nodes[None, :]

            normalized_source_nodes = torch.as_tensor((range(len(source_nodes.unique()))))
            edge_index = torch.cat((normalized_source_nodes.repeat(1, num_nodes),
                                    normalized_source_nodes.repeat(num_nodes, 1).transpose(0, 1).contiguous().view(
                                        (1, num_nodes ** 2))), dim=0)

            # add extra node for classification
            output_nodes = torch.cat((source_nodes[0, :], torch.as_tensor([patient_data.shape[1]])))
            output_nodes = output_nodes[None, :]
            normalized_output_nodes = torch.as_tensor((range(len(output_nodes.unique()))))
            output_edge_index = torch.cat((normalized_output_nodes.repeat(1, num_nodes + 1),
                                           normalized_output_nodes.repeat(num_nodes + 1, 1).transpose(0,
                                                                                                      1).contiguous().view(
                                               (1, (num_nodes + 1) ** 2))), dim=0)

            dem_data = demographic_data[patient_idx, :]

            y = outcomes[patient_idx]
            data.append(Data(x=output_nodes.transpose(0, 1), edge_index=edge_index.long(),
                             output_edge_index=output_edge_index.long(), y=y,
                             demographic=dem_data[None, :]))
        return data


def graph_collate(batch):
    """
    Collate function to use with graph datasets.

    Parameters
    ----------
    batch :

    Returns
    -------

    """
    elem = batch[0]
    if isinstance(elem, Data):
        batch = Batch.from_data_list(batch)
    return batch, batch.y


class SARDData(Dataset):
    """
    Dataset class used for the original SARD implementation.
    """

    def __init__(self, indices, non_temporal, train_indices, outcomes, linear_predictions=None,
                 distill=True):
        """

        Parameters
        ----------
        indices : dict          with train, val and test indices
        outcomes :              outcome labels
        linear_predictions :    predictions from previous model to distill
        distill :               if run for distillation or not, if distillation then get_item returns also predictions
                                of already fit model
        """

        self.distill = distill
        self.outcomes = outcomes
        self.linear_predictions = linear_predictions
        self.indices = indices

        # fix r to py
        non_temporal.rowIdPython = non_temporal.rowIdPython - 1

        # extract age and other covariates
        age_id = 1002
        age_df = non_temporal[non_temporal.covariateId == age_id]
        age_df = age_df.sort_values(by='rowIdPython')
        age = torch.as_tensor(age_df.covariateValue.values, dtype=torch.float32)
        age_squared = age ** 2
        age_sqrt = torch.sqrt(age)
        ages = torch.stack([age, age_squared, age_sqrt]).T
        scaler = StandardScaler()
        scaler.fit(ages[train_indices])
        ages = scaler.transform(ages)

        # other covariates
        other_df = non_temporal[non_temporal.covariateId != age_id].sort_values(by='rowIdPython')
        not_age = torch.zeros((len(ages)))
        not_age[other_df.rowIdPython.values] = torch.as_tensor(other_df.covariateValue.values, dtype=torch.float32)

        self.num = torch.cat([ages, not_age[:, None]], dim=1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        if self.distill:
            return (self.indices[item], self.num[item]), (
                self.outcomes[self.indices[item]], self.linear_predictions[self.indices[item]])
        else:
            return (self.indices[item], self.num[item]), self.outcomes[self.indices[item]]


class VisitSequenceWithLabelDataset(Dataset):
    """
    Dataset class that uses lists of lists
    """

    def __init__(self, seqs, labels, num_features, non_temporal_data, visits, train_indices, reverse=False):
        """
        Args:
        seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
        labels (list): list of labels (int)
        num_features (int): number of total features available
        non_temporal_data (dataframe): dataframe with nonTemporalData such as age or gender.
        visits (list): list of patients with timeId of visits
        train_indices (): indices of training set, used for operations that should only use info from training set
        reverse (bool): If true, reverse the order of sequence (for RETAIN)

        """

        if len(seqs) != len(labels):
            raise ValueError("Sequences and Labels have different lengths")

        # fix r to py
        non_temporal_data.rowIdPython = non_temporal_data.rowIdPython - 1

        # extract age and other covariates
        age_id = 1002
        age_df = non_temporal_data[non_temporal_data.covariateId == age_id]
        age_df = age_df.sort_values(by='rowIdPython')
        age = torch.as_tensor(age_df.covariateValue.values, dtype=torch.float32)
        age_squared = age ** 2
        age_sqrt = torch.sqrt(age)
        ages = torch.stack([age, age_squared, age_sqrt]).T
        scaler = StandardScaler()
        scaler.fit(ages[train_indices])
        ages = torch.as_tensor(scaler.transform(ages), dtype=torch.float32)

        # other covariates
        other_df = non_temporal_data[non_temporal_data.covariateId != age_id].sort_values(by='rowIdPython')
        not_age = torch.zeros((len(seqs)))
        not_age[other_df.rowIdPython.values] = torch.as_tensor(other_df.covariateValue.values, dtype=torch.float32)

        self.train_indices = train_indices
        self.num = torch.cat([ages, not_age[:, None]], dim=1)
        n_visits = [len(v) for v in visits]
        self.max_visits = np.percentile(n_visits, 99).astype(int)
        self.num_features = num_features
        self.visits = torch.vstack(
            [F.pad(torch.as_tensor(v, dtype=torch.long), (0, self.max_visits - len(v))) for v in visits])
        self.seqs = []
        self.lengths = []

        for i, (seq, label) in tqdm(enumerate(zip(seqs, labels))):

            if reverse:
                sequence = list(reversed(seq))
            else:
                sequence = seq

            row = []
            col = []
            val = []
            for j, visit in enumerate(sequence):
                for code in visit:
                    if code < num_features:
                        row.append(j)
                        col.append(code)
                        val.append(1.0)

            if len(sequence) < self.max_visits:
                self.seqs.append(coo_matrix((np.array(val, dtype=np.float32), (np.array(row), np.array(col))),
                                            shape=(self.max_visits, num_features)))
                self.lengths.append(len(sequence))
            else:
                ix = np.array(row) < self.max_visits  # truncate to max visits
                self.seqs.append(
                    coo_matrix((np.array(val, dtype=np.float32)[ix], (np.array(row)[ix], np.array(col)[ix])),
                               shape=(self.max_visits, num_features)))
                self.lengths.append(self.max_visits)

        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.as_tensor(self.seqs[index].todense()), self.num[index, ...], self.labels[index], \
               self.lengths[index], self.visits[index]


class DistillDataset(VisitSequenceWithLabelDataset):
    """
    Dataset class for the distillation where I needed to add the predictions from the teacher model
    """

    def __init__(self, linear_predictions=None, distill=True, **kwargs):
        super(DistillDataset, self).__init__(**kwargs)

        self.distill = distill
        self.linear_predictions = torch.as_tensor(linear_predictions.values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.distill:
            return torch.as_tensor(self.seqs[index].todense()), self.num[index, ...], self.linear_predictions[index], \
                   self.labels[index], \
                   self.lengths[index], self.visits[index]
        else:
            return torch.as_tensor(self.seqs[index].todense()), self.num[index, ...], self.labels[index], \
                   self.lengths[index], self.visits[index]


class RETAIN_dataset(Dataset):
    """
    RETAIN is an RNN and so doesn't need to pad the input but can work with variable length sequences so I used
    this class that doesn't pad the input.
    """

    def __init__(self, seqs, labels, num_features, non_temporal_data, visits, train_indices, reverse=True):
        """
        Args:
        seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
        labels (list): list of labels (int)
        num_features (int): number of total features available
        non_temporal_data (dataframe): dataframe with nonTemporalData such as age or gender.
        visits (list): list of patients with timeId of visits
        train_indices (): indices of training set, used for operations that should only use info from training set
        reverse (bool): If true, reverse the order of sequence (for RETAIN)

        """

        if len(seqs) != len(labels):
            raise ValueError("Sequences and Labels have different lengths")

        # fix r to py
        non_temporal_data.rowIdPython = non_temporal_data.rowIdPython - 1

        # extract age and other covariates
        age_id = 1002
        age_df = non_temporal_data[non_temporal_data.covariateId == age_id]
        age_df = age_df.sort_values(by='rowIdPython')
        age = torch.as_tensor(age_df.covariateValue.values, dtype=torch.float32)
        age_squared = age ** 2
        age_sqrt = torch.sqrt(age)
        ages = torch.stack([age, age_squared, age_sqrt]).T
        age_maxes = torch.max(ages[train_indices], dim=0).values
        ages = ages / age_maxes

        # other covariates
        other_df = non_temporal_data[non_temporal_data.covariateId != age_id].sort_values(by='rowIdPython')
        not_age = torch.zeros((len(seqs)))
        not_age[other_df.rowIdPython.values] = torch.as_tensor(other_df.covariateValue.values, dtype=torch.float32)

        self.num = torch.cat([ages, not_age[:, None]], dim=1)
        self.visits = visits
        self.seqs = []
        self.lengths = []

        for i, (seq, label) in enumerate(zip(seqs, labels)):

            if reverse:
                sequence = list(reversed(seq))
            else:
                sequence = seq

            row = []
            col = []
            val = []
            for j, visit in enumerate(sequence):
                for code in visit:
                    if code < num_features:
                        row.append(j)
                        col.append(code)
                        val.append(1.0)

            self.seqs.append(coo_matrix((np.array(val, dtype=np.float32), (np.array(row), np.array(col))),
                                        shape=(len(sequence), num_features)))
            self.lengths.append(len(sequence))
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.as_tensor(self.seqs[index].todense()), self.num[index, ...], self.labels[index], \
               self.lengths[index], self.visits[index]


def pad(batch):
    """
    Collate function that I use with RETAIN and the vanilla Transformer.

    Parameters
    ----------
    batch :

    Returns
    -------

    """
    batch_split = list(zip(*batch))
    seqs, num, targs, lengths, visits = batch_split[0], batch_split[1], batch_split[2], batch_split[3], batch_split[4]
    num = torch.vstack([torch.as_tensor(sample, dtype=torch.float32) for sample in zip(*num)]).T
    visits = [torch.as_tensor(s, dtype=torch.long) for s in visits]
    return [list(seqs), num, torch.as_tensor(lengths, dtype=torch.long), visits], \
           torch.as_tensor(targs, dtype=torch.float32)


def distill_pad(batch):
    """
    Collate function I use when distilling

    Parameters
    ----------
    batch :

    Returns
    -------

    """
    batch_split = list(zip(*batch))
    seqs, num, preds, targs, lengths, visits = batch_split[0], batch_split[1], batch_split[2], batch_split[3], \
                                               batch_split[4], batch_split[5]
    num = torch.vstack([torch.as_tensor(sample, dtype=torch.float32) for sample in zip(*num)]).T
    visits = [torch.as_tensor(s, dtype=torch.long) for s in visits]
    return [list(seqs), num, torch.as_tensor(lengths, dtype=torch.long), visits], \
           [torch.as_tensor(targs, dtype=torch.float32), torch.as_tensor(preds, dtype=torch.float32)]
