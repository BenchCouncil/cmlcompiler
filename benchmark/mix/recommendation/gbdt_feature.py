from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

class GradientBoostingFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Using GBDT to get feature vector
    """
    def __init__(
        self,
        stack_to_X=True,
        **kwargs,
    ):
        # Deciding whether to append features or simply return generated features
        self.stack_to_X = stack_to_X
        self.gbm = GradientBoostingClassifier(**kwargs)

    def _get_leaves(self, X):
        X_leaves = self.gbm.apply(X)
        n_rows, n_cols, _ = X_leaves.shape
        X_leaves = X_leaves.reshape(n_rows, n_cols)
        return X_leaves

    def _decode_leaves(self, X):
        return self.encoder.transform(X).todense()

    def fit(self, X, y):
        self.gbm.fit(X, y)
        self.encoder = OneHotEncoder(categories="auto")
        X_leaves = self._get_leaves(X)
        self.encoder.fit(X_leaves)
        return self

    def transform(self, X):
        R = self._decode_leaves(self._get_leaves(X))
        X_new = np.hstack((X, R)) if self.stack_to_X == True else R
        return X_new

