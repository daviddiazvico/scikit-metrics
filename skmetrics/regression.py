"""
Scikit-learn-compatible metrics for regression problems.

@author: David Diaz Vico
@license: MIT
"""

import numpy as np


def relative_mean_absolute_error(y_true, y_pred, sample_weight=None):
    """ Relative mean absolute error

        Calculates the relative mean absolute error:
        100 * |y_true - y_pred| / |y_true|.

        Parameters
        ----------
        y_true: array-like, shape = [n_samples]
                Ground truth (correct) target values.
        y_pred: array-like, shape = [n_samples]
                Estimated targets as returned by a classifier.
        sample_weight: array-like of shape = [n_samples], optional
                       Sample weights.

        Returns
        -------
        rmae: float
              Relative mean absolute error.
    """
    diff = np.abs(y_pred - y_true) / np.abs(y_true)
    rmae = np.average(diff, weights=sample_weight, axis=0)
    return rmae
