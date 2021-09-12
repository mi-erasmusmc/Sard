<<<<<<< HEAD
"""
Linear model from github.com/clinicalml/omop-learn
"""

import numpy as np
import scipy.sparse
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
=======
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import scipy.sparse
import numpy as np
>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66


def sparse_ufunc(f):
    def wrapper(*a, **k):
        X = a[0]
        if not scipy.sparse.isspmatrix(X):
            raise ValueError
        X2 = X.copy()
        X2.data = f(X2.data, *(a[1:]), **k)
        return X2
<<<<<<< HEAD

=======
>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66
    return wrapper


@sparse_ufunc
def tr_func(X, kwarg=1):
    return np.clip(X, 0, kwarg)

<<<<<<< HEAD

=======
>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66
func = FunctionTransformer(
    func=tr_func,
    accept_sparse=True,
    validate=True,
    kw_args={'kwarg': 1}
)


<<<<<<< HEAD
def gen_lr_pipeline(C=0.01, class_weight=None, solver='liblinear'):
    lr = LogisticRegression(
        class_weight=class_weight, C=C,
        penalty='l1', fit_intercept=True,
        solver=solver, random_state=0,
        verbose=0, max_iter=200, tol=1e-4
=======
def gen_lr_pipeline(C=0.01):
    lr = LogisticRegression(
        class_weight='balanced', C=C,
        penalty='l1', fit_intercept=True,
        solver='liblinear', random_state=0,
        verbose=0, max_iter=200, tol=1e-1
>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66
    )

    # The classifier will transform each data point using func, which here takes a count vector to a binary vector
    # Then, it will use logistic regression to classify the transformed data
    clf_lr = Pipeline([
<<<<<<< HEAD
        ('func', func),
        ('lr', lr)
    ])
    return clf_lr
    # all_preds
=======
        ('func',func),
        ('lr', lr)
    ])
    return clf_lr
    # all_preds
>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66
