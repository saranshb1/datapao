
import psutil
import pandas as pd

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

