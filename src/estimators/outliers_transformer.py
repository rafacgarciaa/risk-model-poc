from typing import List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OutliersTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, outlier_identifiers: List[int]):
        self.outlier_identifier = outlier_identifiers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        replace_dict = {str(outlier): np.nan for outlier in self.outlier_identifier}
        X["DAYS_EMPLOYED"].replace(replace_dict, inplace=True)
        return X
