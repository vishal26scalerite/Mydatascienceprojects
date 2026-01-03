import numpy as np
import pandas as pd
from Aerofit_Case_Study import utils


def make_sample_df():
    data = {
        'Product': ['KP281', 'KP281', 'KP481'],
        'Age': [25, 25, 30],
        'Gender': ['Male', 'Male', 'Female'],
        'Education': [16, 16, 15],
        'MaritalStatus': ['Single', 'Single', 'Partnered'],
        'Usage': [3, 3, 4],
        'Fitness': [3, 3, 4],
        'Income': [50000, 50000, 60000],
        'Miles': [90, 90, 100],
    }
    return pd.DataFrame(data)


def test_basic_checks_and_duplicates():
    df = make_sample_df()
    summary = utils.basic_checks(df)
    assert summary['shape'] == (3, 9)
    assert summary['duplicates'] == 1
    assert 'Age' in summary['dtypes']
    assert isinstance(summary['nulls']['Age'], (int, np.integer))


def test_preprocess_create_feature_matrix():
    df = make_sample_df()
    X, y, dfc = utils.preprocess(df, drop_duplicates=True)
    # duplicate rows should be dropped (two identical rows -> 2 unique rows)
    assert dfc.shape[0] == 2
    assert X.shape[0] == 2
    # numeric columns present
    for col in ['Age', 'Education', 'Usage', 'Fitness', 'Income', 'Miles']:
        assert col in X.columns
    # categorical dummies exist
    assert any(col.startswith('Gender_') for col in X.columns)
    assert any(col.startswith('MaritalStatus_') for col in X.columns)
    # y values correspond to products in cleaned df
    assert set(y.unique()).issubset(set(dfc['Product'].unique()))
