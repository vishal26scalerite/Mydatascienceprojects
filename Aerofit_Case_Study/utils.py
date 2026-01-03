import pandas as pd
import numpy as np

DATA_URL = 'https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/001/125/original/aerofit_treadmill.csv?1639992749'
RANDOM_STATE = 42


def load_data(url: str = DATA_URL) -> pd.DataFrame:
    """Load the Aerofit dataset from a URL and return a DataFrame."""
    return pd.read_csv(url)


def basic_checks(df: pd.DataFrame) -> dict:
    """Return a small summary of the DataFrame useful for tests and quick checks.

    The returned dict contains: shape, nulls (as dict), duplicates (int), and dtypes (as dict).
    """
    return {
        'shape': df.shape,
        'nulls': df.isnull().sum().to_dict(),
        'duplicates': int(df.duplicated().sum()),
        'dtypes': df.dtypes.astype(str).to_dict(),
    }


def preprocess(df: pd.DataFrame, drop_duplicates: bool = True):
    """Return (X, y, df_clean) where X is the feature matrix and y is the target Series.

    - Ensures expected dtypes for categorical columns.
    - Drops duplicates if requested.
    - Numeric features: Age, Education, Usage, Fitness, Income, Miles
    - Categorical features one-hot encoded: Gender, MaritalStatus (drop_first=True)
    """
    dfc = df.copy()
    if drop_duplicates:
        dfc = dfc.drop_duplicates().reset_index(drop=True)

    # Ensure categorical columns are strings
    for c in ['Product', 'Gender', 'MaritalStatus']:
        if c in dfc.columns:
            dfc[c] = dfc[c].astype(str)

    num_cols = ['Age', 'Education', 'Usage', 'Fitness', 'Income', 'Miles']
    for c in num_cols:
        if c not in dfc.columns:
            raise KeyError(f"Expected numeric column '{c}' not found in DataFrame")

    X_num = dfc[num_cols].copy()
    X_cat = pd.get_dummies(dfc[['Gender', 'MaritalStatus']], drop_first=True)
    X = pd.concat([X_num, X_cat], axis=1)
    y = dfc['Product'].reset_index(drop=True)
    return X, y, dfc


# convenience alias
create_feature_matrix = preprocess
