from typing import List

CATEGORICAL_COLUMNS = [
    "ORGANIZATION_TYPE",
    "OCCUPATION_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_TYPE_SUITE",
    "WEEKDAY_APPR_PROCESS_START",
    "WALLSMATERIAL_MODE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "NAME_EDUCATION_TYPE",
    "FONDKAPREMONT_MODE",
    "CODE_GENDER",
    "HOUSETYPE_MODE",
]

BINARY_CATEGORICAL_COLUMNS = [
    "NAME_CONTRACT_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "EMERGENCYSTATE_MODE",
]


def exclude_categorical_columns(columns: List[str]) -> List[str]:
    return [
        c
        for c in columns
        if not (c in CATEGORICAL_COLUMNS or c in BINARY_CATEGORICAL_COLUMNS)
    ]
