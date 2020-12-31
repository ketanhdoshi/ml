# --------------------------------------------------------------------------------------------
# Utility functions to pre-process data
# --------------------------------------------------------------------------------------------

import pandas as pd

# --------------------------------------------------------------------------------------------
# Get the count of missing values in each column of the dataframe
# --------------------------------------------------------------------------------------------
def get_missing_value_counts(df):
    return (df.isnull().sum())

# --------------------------------------------------------------------------------------------
# Get the underlying numpy array from a dataframe
# --------------------------------------------------------------------------------------------
def get_nparray(df):
    return (df.values)

# --------------------------------------------------------------------------------------------
# Remove rows or columns which contain missing values
# --------------------------------------------------------------------------------------------
def remove_missing_rowcolumns(df, remove_row = True, col_subset=None):
    if (remove_row):
        if col_subset is None:
            # only drop rows where NaN appear in specific columns
            return (df.dropna(axis=0))
        else:
            # remove all rows that contain missing values
            return (df.dropna(subset=col_subset))
    else:
        # remove all columns that contain missing values
        return (df.dropna(axis=1))

# --------------------------------------------------------------------------------------------
# Impute and fill in missing values
# --------------------------------------------------------------------------------------------
from sklearn.preprocessing import Imputer
def impute_missing_values (df):
    imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imr = imr.fit(df)
    imp_df = imr.transform(df.values)
    return pd.DataFrame(imp_df, index = df.index, columns = df.columns)

# --------------------------------------------------------------------------------------------
# Map Encode Ordinal Categorical features
# The existing categorical column is modified in-place
# --------------------------------------------------------------------------------------------
def encode_cat_ordinal (df, cat_col, mapping):
    df[cat_col] = df[cat_col].map(mapping)
