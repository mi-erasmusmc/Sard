import numpy as np
import scipy
import sparse


def window_data_sorted(window_lengths=None, feature_matrix=None, all_feature_names=None):
    """
    Takes in a sparse feature matrix and windows and counts features in windows.

    Adapted from github.com/clinicalml/omop-learn

    Parameters
    ----------
    window_lengths :            A list of window lengths
    feature_matrix :            Pytorch sparse COO matrix, num Patients X num Features X num Timepoints
    all_feature_names :

    Returns
    -------
    feature_matrix_counts :     Matrix with counts in each window
    feature_names :             Feature names of the windowed features (name of window added)

    """
    all_times = np.arange(feature_matrix.size()[2])  # last axis is time

    windowed_time_ixs = dict()
    for ix, interval in enumerate(window_lengths):
        windowed_time_ixs[interval] = all_times[-1] - interval, all_times[-1]

    feature_matrix_slices = []
    feature_names = []

    # since pytorch sparse matrices don't support indexing I convert it to pydata sparse format
    feature_matrix = sparse.COO(coords=feature_matrix.indices(), data=feature_matrix.values(), shape=feature_matrix.size())
    for interval in sorted(windowed_time_ixs):
        end_time = windowed_time_ixs[interval][1]
        start_time = windowed_time_ixs[interval][0]
        feature_matrix_slices.append(
            feature_matrix[:, :, start_time:end_time])
        feature_names += ['{} - {} days'.format(n, interval) for i, n in enumerate(all_feature_names)]
    feature_matrix_counts = scipy.sparse.vstack(
        [
            m.sum(axis=2).T.tocsr()
            for m in feature_matrix_slices
        ]
    )
    return feature_matrix_counts, feature_names
