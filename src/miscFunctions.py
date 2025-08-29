
import psutil
import pandas as pd
import numpy as np

def ram_usage():
    """
    Returns current RAM usage (used/total) in MB and percentage.
    """
    mem = psutil.virtual_memory()
    used_mb = mem.used / (1024 ** 2)
    total_mb = mem.total / (1024 ** 2)
    percent = mem.percent
    
    """
    Only call the function in line as ram_usage() in any sequence.
    """

    print(f"RAM Used: {used_mb:.2f} MB / {total_mb:.2f} MB ({percent}%)")
    #return mem.percent  # Returning percentage for programmatic use


def getNonNumericCols(df: pd.DataFrame):
    """
    Returns a list of column names in the DataFrame
    that are not numeric (includes object, category, bool, datetime, etc.).
    """
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()

    """
    Way to use:

    non_num_cols = getNonNumericCols(df)
    """    

    return non_numeric_cols


def getDataFrameDTypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the datatype of each column in the DataFrame.

    Args:
        df (pd.DataFrame): Input pandas DataFrame.

    Returns:
        pd.DataFrame: DataFrame with columns and their datatypes.
    """
    dtype_df = pd.DataFrame(df.dtypes, columns=["dtype"]).reset_index()
    dtype_df.columns = ["column", "dtype"]

    """
    Way to use function ->
    
    dtypes_cols = getDataFrameDTypes(df)
    """

    return dtype_df


def missingPercentage(df, extra_missing_values=None):
    """
    Calculates missing value percentage for each column.
    
    Parameters:
    - df: pandas DataFrame
    - extra_missing_values: list of values to treat as missing 
      (e.g., ['-', 'NA', 'N/A', ''])
    
    Returns:
    DataFrame with columns:
    - 'missing_count'
    - 'missing_percent'
    """
    df_copy = df.copy()

    # Replace extra placeholders with NaN
    if extra_missing_values is not None:
        df_copy.replace(extra_missing_values, np.nan, inplace=True)

    total_rows = len(df_copy)
    missing_count = df_copy.isnull().sum()
    missing_percent = (missing_count / total_rows) * 100

    result = pd.DataFrame({
        'missing_count': missing_count,
        'missing_percent': missing_percent
    }).sort_values(by='missing_percent', ascending=False)

    return result

