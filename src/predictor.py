from typing import List

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from src.config import OUTLIER_IDS
from src.estimators.outliers_transformer import OutliersTransformer
from src.utils.columns import (
    BINARY_CATEGORICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    exclude_categorical_columns,
)


def build_transformer_pipeline(df_columns: List[str]) -> Pipeline:
    binary_categorical_preprocessor = Pipeline(
        steps=[
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    categorical_preprocessor = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_preprocessor = Pipeline(
        steps=[
            ("imputation_mean", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scaler", MinMaxScaler()),
        ]
    )
    data_transformer = ColumnTransformer(
        [
            (
                "binary_categorical",
                binary_categorical_preprocessor,
                BINARY_CATEGORICAL_COLUMNS,
            ),
            ("categorical", categorical_preprocessor, CATEGORICAL_COLUMNS),
            (
                "numerical",
                numeric_preprocessor,
                exclude_categorical_columns(df_columns),
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("fix_outliers", OutliersTransformer(outlier_identifiers=OUTLIER_IDS)),
            ("transform_data", data_transformer),
        ]
    )


def build_classifier_pipeline(df_columns: List[str]) -> Pipeline:
    return make_pipeline(
        build_transformer_pipeline(df_columns),
        LogisticRegression(C=0.0001, max_iter=500),
    )
