# A python function to handle outliers and the method used was capping(Winsorization)
#This method keeps every row, reduces extreme impact and preserves distribution shape of the our data

#step importing all the necessary libraries.
import pandas as pd
import numpy as np
from pathlib import Path

#defining iqr_winsorize function
def iqr_winsorize(df: pd.DataFrame,cols: list,factor: float = 1.5,verbose: bool = True) -> pd.DataFrame:
   
    df_out = df.copy()
    capped_counts = {}

    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        # Count how many will be capped
        n_low  = (df[col] < lower).sum()
        n_high = (df[col] > upper).sum()
        capped_counts[col] = n_low + n_high

        # Apply caps
        df_out[col] = df_out[col].clip(lower=lower, upper=upper)

    if verbose:
        print("=== WINSORIZED OUTLIERS (capped) ===")
        for c, n in capped_counts.items():
            print(f"{c}: {n} values capped")
    return df_out