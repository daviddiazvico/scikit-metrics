"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import numpy as np

from skmetrics import separability_score


def test_separability_score():
    """Tests separability score."""
    x = np.random.rand(4, 3)
    y = np.array([0, 0, 1, 1])
    separability_score(y_true=y, y_pred=x)
