
#import pandas as pd
#import polars as pl
#import numpy as np
import glob
import os
import time
import psutil
#import gc
#import joblib
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import input_file_name, regexp_extract


def readCombineCSVSafe(folder_path, prefix, max_memory_fraction=0.1):
    """
    Reads CSVs into a Spark or pandas DataFrame, measures load time,
    estimates memory usage, and adds a 'source_file' column with the file name.

    Parameters:
        folder_path (str): Path to folder containing CSV files.
        prefix (str): Prefix of files to match (e.g., "ACY").
        max_memory_fraction (float): Max fraction of available RAM for pandas conversion.

    Returns:
        tuple: (DataFrame, load_time_seconds)
    """
    spark = SparkSession.builder \
        .appName("Timed CSV Loader with Memory Check + Filename") \
        .master("local[*]") \
        .getOrCreate()

    # File matching pattern
    file_pattern = f"{folder_path}/{prefix}*.csv"
    csv_files = glob.glob(file_pattern)

    if not csv_files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

    start_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)  # MB

    # Read CSVs and add source file path column
    df_spark = spark.read.csv(file_pattern, header=True, inferSchema=True) \
        .withColumn("source_file_path", input_file_name())

    # Extract only the file name from the path
    df_spark = df_spark.withColumn(
        "source_file",
        regexp_extract("source_file_path", r"([^/\\]+$)", 1)
    )

    total_rows = df_spark.count()
    total_columns = len(df_spark.columns)

    # Estimate dataset memory size
    sample_pdf = df_spark.limit(1000).toPandas()
    sample_size_bytes = sample_pdf.memory_usage(deep=True).sum()
    estimated_size_bytes = (sample_size_bytes / len(sample_pdf)) * total_rows

    available_mem_bytes = psutil.virtual_memory().available
    safe_limit_bytes = available_mem_bytes * max_memory_fraction

    load_time = time.time() - start_time
    mem_after = process.memory_info().rss / (1024 ** 2)  # MB

    # Print useful diagnostics
    print(f"✅ Loaded {len(csv_files)} files into Spark DataFrame")
    print(f"📊 Shape: {total_rows} rows × {total_columns} columns")
    print(f"⏱ Load time: {load_time:.2f} seconds")
    print(f"💾 Memory before: {mem_before:.2f} MB")
    print(f"💾 Memory after:  {mem_after:.2f} MB")
    print(f"📦 Estimated dataset size (pandas): {estimated_size_bytes / (1024**2):.2f} MB")
    print(f"💻 Available memory limit: {safe_limit_bytes / (1024**2):.2f} MB")

    """
    
    Way to use the function

    data_path = "drive/folder/data"
    df, load_time = readCombineCSVSafe(data_path, "prefix*")

    """

    # Decide whether to convert to pandas or keep Spark DataFrame
    if estimated_size_bytes <= safe_limit_bytes:
        print(f"✅ Converting to pandas DataFrame")
        return df_spark.toPandas(), load_time
    else:
        print(f"⚠ Too large for pandas. Returning Spark DataFrame.")
        return df_spark, load_time


