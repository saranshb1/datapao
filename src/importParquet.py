
import dask.dataframe as dd

def loadParquetDataset(
    filepath, 
    engine="pyarrow", 
    to_pandas=True, 
    columns=None
):
    """
    Load a partitioned Parquet dataset into a Dask or Pandas DataFrame.
    
    Parameters
    ----------
    filepath : str
        Path to the Parquet dataset (folder or file).
    
    engine : str, default="pyarrow"
        Parquet engine to use ("pyarrow" or "fastparquet").
    
    to_pandas : bool, default=True
        If True, converts the Dask DataFrame to a Pandas DataFrame.
        If False, returns the Dask DataFrame.
    
    columns : list of str, optional
        Subset of columns to load. If None, load all columns.
    
    Returns
    -------
    ddf or df : dask.DataFrame or pandas.DataFrame
        The loaded dataset.
    """
    ddf = dd.read_parquet(filepath, engine=engine, columns=columns)
    return ddf.compute() if to_pandas else ddf

