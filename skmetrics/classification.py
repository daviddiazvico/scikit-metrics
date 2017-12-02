"""
Scikit-learn-compatible metrics for classification problems.

@author: David Diaz Vico
@license: MIT
"""

import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.metrics.cluster.supervised import (check_clusterings,
                                                contingency_matrix)
from sklearn.preprocessing import OneHotEncoder


def g_score(y_true, y_pred, eps=None, sparse=False):
    """ G-score

        Calculates the G-score: sqrt(prod(true_rates)).

        Parameters
        ----------
        y_true: array-like, shape = [n_samples]
                Ground truth (correct) target values.
        y_pred: array-like, shape = [n_samples]
                Estimated targets as returned by a classifier.
        eps: None or float, optional.
             If a float, that value is added to all values in the contingency
             matrix. This helps to stop NaN propagation. If ``None``, nothing is
             adjusted.
        sparse: boolean, optional.
                If True, return a sparse CSR continency matrix. If ``eps is not
                None``, and ``sparse is True``, will throw ValueError.

        Returns
        -------
        g: float
           G-score.
    """
    y_true, y_pred = check_clusterings(y_true, y_pred)
    c = contingency_matrix(y_true, y_pred, eps=eps, sparse=sparse)
    d = c.diagonal()
    true_rates = d.reshape([len(d), 1]) / c.sum(axis=1).ravel()
    g = np.prod(true_rates)**(1.0 / c.shape[0])
    return g


def geometric_roc_auc_score(y_true, y_score, pos_label=None, sample_weight=None,
                            drop_intermediate=True, reorder=False):
    """ Multiclass Geometric ROC AUC score

        Calculates the multiclass geometric mean ROC AUC score:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html.

        Parameters
        ----------
        y_true: array-like, shape = [n_samples]
                Ground truth (correct) target values.
        y_pred: array-like, shape = [n_samples]
                Estimated targets as returned by a classifier.
        pos_label: int or str, default=None
                   Label considered as positive and others are considered
                   negative.
        sample_weight: array-like of shape = [n_samples], optional
                       Sample weights.
        drop_intermediate: boolean, optional (default=True)
                           Whether to drop some suboptimal thresholds which
                           would not appear on a plotted ROC curve. This is
                           useful in order to create lighter ROC curves.
        reorder: boolean, optional (default=False)
                 If True, assume that the curve is ascending in the case of
                 ties, as for an ROC curve. If the curve is non-ascending, the
                 result will be wrong.

        Returns
        -------
        gaur: float
              Multiclass Geometric ROC AUC score.
    """
    n_classes = len(np.unique(y_true))
    y_true = OneHotEncoder().fit_transform(y_true.reshape((len(y_true),
                                                           1))).todense()
    classes_auc = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i],
                                pos_label=pos_label,
                                sample_weight=sample_weight,
                                drop_intermediate=drop_intermediate)
        classes_auc.append(auc(fpr, tpr, reorder=reorder))
    gaur = np.prod(np.asarray(classes_auc))**(1./n_classes)
    return gaur


def _scatter_matrices(y_true, y_pred):
    """Within and total scatter matrices."""
    sc = [np.cov(m=y_pred[np.where(y_true.T == c)[0]], rowvar=0)
          for c in np.unique(y_true.T)]
    sw = np.mean(sc, axis=0)
    st = np.cov(m=y_pred, rowvar=0)
    return sw, st


def separability_score(y_true, y_pred):
    """ Separability score

        Calculates the separability score.

        Parameters
        ----------
        y_true: array-like, shape = [n_samples]
                Ground truth (correct) target values.
        y_pred: array-like, shape = [n_samples]
                Projections returned by a transformer.

        Returns
        -------
        s: float
           Separability score.
    """
    sw, st = _scatter_matrices(y_true=y_true, y_pred=y_pred)
    s = st / sw
    if not np.isscalar(s):
        s = np.trace(s)
    return s
