import pytest
from io import StringIO
import pandas as pd
import pandas.util.testing as pdt
import numpy.testing as npt 
from data_prep import get_missing_value_counts, get_nparray, remove_missing_rowcolumns, impute_missing_values, encode_cat_ordinal

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

@pytest.fixture
def mydf():
    return pd.read_csv(StringIO(csv_data))

def catdf():
    df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])

    df.columns = ['color', 'size', 'price', 'classlabel']
    return(df)

def test_missing_values():
    datadf = mydf()
    miss_counts = get_missing_value_counts(datadf)
    expected = pd.Series(data = [0, 0, 1, 1], index=['A','B','C','D'])
    pdt.assert_series_equal(miss_counts, expected)

def test_nparray():
    datadf = mydf()
    nparr = get_nparray(datadf)
    expected = [[ 1.,  2.,  3.,  4.],
       [ 5.,  6., float('NaN'),  8.],
       [10., 11., 12., float('NaN')]]
    npt.assert_array_equal(nparr, expected)

def test_remove_missing_rowcolumns():
    datadf = mydf()
    rowdf = remove_missing_rowcolumns(datadf)
    expected = pd.DataFrame([{'A': 1., 'B': 2., 'C': 3. ,'D': 4.}])
    pdt.assert_frame_equal(rowdf, expected)

    coldf = remove_missing_rowcolumns(datadf, False)
    expected = pd.DataFrame({'A': [1., 5., 10.], 'B': [2., 6., 11.]})
    pdt.assert_frame_equal(coldf, expected)
				
    row_subset_df = remove_missing_rowcolumns(datadf, True, col_subset=['C'])
    expected = pd.DataFrame(data = {'A': [1., 10.], 'B': [2., 11.], 
                  'C': [3., 12.], 'D': [4., float('NaN')]}, index=[0, 2])
    pdt.assert_frame_equal(row_subset_df, expected)

def test_impute_missing_values():
    datadf = mydf()
    imputed_df = impute_missing_values (datadf)
    expected = pd.DataFrame(data = {'A': [1., 5., 10.], 'B': [2., 6., 11.], 
                  'C': [3., 7.5, 12.], 'D': [4., 8., 6.]}, index=[0, 1, 2])
    pdt.assert_frame_equal(imputed_df, expected)

def test_encode_cat_ordinal():
    datadf = catdf()
    size_mapping = {'XL': 3, 'L': 2, 'M': 1}
    encode_cat_ordinal (datadf, 'size', size_mapping)
    expected = pd.DataFrame([['green', 1, 10.1, 'class1'],
                   ['red', 2, 13.5, 'class2'],
                   ['blue', 3, 15.3, 'class1']])
    expected.columns = ['color', 'size', 'price', 'classlabel']
    pdt.assert_frame_equal(datadf, expected)