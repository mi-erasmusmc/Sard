import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.utils.sparsefuncs import min_max_axis


class MinMaxScaler(TransformerMixin, BaseEstimator):
    """
    If I wanted to use min-max scaling on sparse data I rewrote the scikit learn min-max scaler. Didn't end up using it.
    """

    def __init__(self, feature_range=(0, 1), copy=True, clip=False):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.min_
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def fit(self, X, y):
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y):
        feature_range = self.feature_range

        first_pass = not hasattr(self, 'n_samples_seen_')
        data_min, data_max = min_max_axis(X, axis=0)

        if first_pass:
            self.n_samples_seen_ = X.shape[0]
        else:
            data_min = np.minimum(self.data_min_, data_min)
            data_max = np.maximum(self.data_max_, data_max)
            self.n_samples_seen_ += X.shape[0]

        data_range = data_max - data_min
        self.scale_ = ((feature_range[1] - feature_range[0]) / _handle_zeros_in_scale(data_range))
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        if isinstance(X, sparse.csc_matrix):
            X.data *= np.repeat(self.scale_, np.diff(X.indptr))
            X.data += np.repeat(self.min_, np.diff(X.indptr))
        elif isinstance(X, sparse.csr_matrix):
            X.data *= self.scale_.take(X.indices, mode='clip')
            X.data += self.min_.take(X.indices, mode='clip')
        if self.clip:
            np.clip(X, self.feature_range[0], self.feature_range[1], out=X)
        return X

    def inverse_transform(self, X):
        X -= self.min_
        X /= self.scale_
        return X
