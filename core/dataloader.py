import pandas as pd

def load_data(path):
    df = pd.read_csv(path, header=None)

    if df.shape[1] != 2:
        raise ValueError("CSV must have exactly 2 columns (x,y)")

    return df.values
