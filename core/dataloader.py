import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)

    if df.shape[1] != 2:
        raise ValueError("CSV must contain exactly 2 columns: x,y")

    if not np.issubdtype(df.dtypes.iloc[0], np.number) or \
       not np.issubdtype(df.dtypes.iloc[1], np.number):
        raise ValueError("CSV must contain only numeric x,y values")

    return df.values
