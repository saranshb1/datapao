import dask.dataframe as dd
from dask.diagnostics import ProgressBar

def saveToParquet(
    df, 
    filename="finalData.parquet", 
    npartitions=1, 
    compression="snappy", 
    engine="pyarrow", 
    write_index=False,
    show_progress=True
):
    """
    Save a pandas DataFrame to a single Parquet file using Dask.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to be saved.
    
    filename : str, default="finalData.parquet"
        Output parquet file path.
    
    npartitions : int, default=1
        Number of partitions for Dask DataFrame.
        (1 ensures a single output parquet file.)
    
    compression : str, default="snappy"
        Compression type ("snappy", "gzip", "brotli", etc.).
    
    engine : str, default="pyarrow"
        Parquet engine to use ("pyarrow" or "fastparquet").
    
    write_index : bool, default=False
        Whether to include the DataFrame index in the parquet file.
    
    show_progress : bool, default=True
        Whether to show a progress bar while writing.
    
    Returns
    -------
    None
        Writes the parquet file to disk.
    """
    # Convert pandas DataFrame to Dask DataFrame
    ddf = dd.from_pandas(df, npartitions=npartitions)
    
    # Write with optional progress bar
    if show_progress:
        with ProgressBar():
            ddf.to_parquet(
                filename, 
                compression=compression, 
                engine=engine, 
                write_index=write_index
            )
    else:
        ddf.to_parquet(
            filename, 
            compression=compression, 
            engine=engine, 
            write_index=write_index
        )
