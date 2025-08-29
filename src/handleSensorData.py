
import pandas as pd
import numpy as np

def handleMissingSensorData(df: pd.DataFrame, col: str, method: str = 'auto') -> pd.DataFrame:
    """
    Handle missing sensor data in a DataFrame column using different strategies.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the sensor data
    col : str
        Column name to process
    method : str, default='auto'
        Method for handling missing data:
        - 'auto'         : Forward-fill short gaps (limit=5), then linear interpolate
        - 'ffill'        : Forward fill, fallback to backward fill
        - 'interpolate'  : Linear interpolation both directions

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled in the specified column
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")

    if method == 'auto':
        # Forward-fill up to 5 consecutive missing values
        df[col] = df[col].ffill(limit=5)
        # Interpolate for medium gaps
        df[col] = df[col].interpolate(method='linear', limit_direction='both')

    elif method == 'ffill':
        # Fill forward, then backward for any remaining NaNs
        df[col] = df[col].ffill().bfill()

    elif method == 'interpolate':
        # Linear interpolation in both directions
        df[col] = df[col].interpolate(method='linear', limit_direction='both')

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from ['auto', 'ffill', 'interpolate'].")

    return df


def removeOutliersIQR(df: pd.DataFrame, column: str):
    """
    Detects outliers in a DataFrame column using the IQR method 
    and replaces them with NaN (to allow interpolation later).
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers replaced by NaN
    """
    # Calculate Q1 and Q3
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Replace outliers with NaN
    df[column] = df[column].where(
        (df[column] >= lower_bound) & (df[column] <= upper_bound),
        np.nan
    )
    
    return df


