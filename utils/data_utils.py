import numpy as np
import scipy
import sparse


def add_age_gender(feature_matrix, nonTemporalData, feature_names, age_normalized=True,
                   age_squared=True, age_sqrt=True):
    """Adds age and gender to 2d feature matrix n_patients x n_features

    Parameters
    ----------
    feature_matrix :            scipy csc sparse matrix object after windowing of size n_patients x n_features
    nonTemporalData :           The non temporal data in a dataframe
    feature_names :             names of the features, need to add age and gender
    age_normalized : bool       if age variable has been normalized to be between 0 and 1
    age_squared : bool          if age squared should be added to account for supra-linear effects
    age_sqrt : bool             if sqrt of age should be added to account for sub-linear effects

    Returns
    -------
    feature_matrix :     input with two more columns, one with age and one with gender
    """

    if age_normalized:
        age_covariate_id = nonTemporalData[(nonTemporalData.timeId.isna()) & (nonTemporalData.covariateValue < 1.0) &
                                           (nonTemporalData.covariateValue > 0.0)].covariateId.unique()[0]
    else:
        age_covariate_id = \
            nonTemporalData[
                (nonTemporalData.timeId == -1) & (nonTemporalData.covariateValue > 1.0)].covariateId.unique()[0]

    nonTemporalData = nonTemporalData.sort_values(by='rowIdPython')

    covariate_ids = nonTemporalData.covariateId.unique()
    gender_covariate_id = covariate_ids[covariate_ids != age_covariate_id][0]

    only_age = nonTemporalData[nonTemporalData['covariateId'] == age_covariate_id]['covariateValue'].values
    if age_squared & age_sqrt:
        age_squared_data = only_age ** 2
        age_sqrt_data = np.sqrt(only_age)
        age = np.stack((only_age, age_squared_data, age_sqrt_data))
    else:
        age = only_age

    gender = np.zeros_like(only_age)
    gender_index = (nonTemporalData['covariateId'] == gender_covariate_id)
    subject_index = nonTemporalData[gender_index].rowIdPython.values - 1
    gender[subject_index] = 1
    age_gender = np.vstack((age, gender)).T
    age_gender_sparse = scipy.sparse.csc_matrix(age_gender, dtype='float')

    feature_names.append('age')
    if age_squared:
        feature_names.append('age_squared')
    if age_sqrt:
        feature_names.append('age_sqrt')
    feature_names.append('gender = Male')

    feature_matrix = scipy.sparse.hstack((feature_matrix, age_gender_sparse))
    return feature_matrix, feature_names


def window_data_sorted(window_lengths=None, feature_matrix=None, all_feature_names=None):
    """
    Takes in a sparse feature matrix and windows and counts features in windows.

    Adapted from github.com/clinicalml/omop-learn

    Parameters
    ----------
    window_lengths :            A list of window lengths
    feature_matrix :            Pytorch sparse COO matrix, num Patients X num Features X num timepoints
    all_feature_names :

    Returns
    -------
    feature_matrix_counts :     Matrix with counts in each window
    feature_names :             Feature names of the windowed features (name of window added)

    """

    feature_matrix_slices = []
    feature_names = []

    # since pytorch sparse matrices don't support indexing I convert it to pydata sparse format
    feature_matrix = sparse.COO(coords=feature_matrix.indices(), data=feature_matrix.values(),
                                shape=feature_matrix.size())
    for interval in sorted(window_lengths):
        end_time = interval
        start_time = 1
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
