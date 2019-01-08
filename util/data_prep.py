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



