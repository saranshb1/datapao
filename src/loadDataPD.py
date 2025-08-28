
import os
import glob
import time
import psutil
import pandas as pd

def loadCSVToDataFrame(path, prefix: str = "defined_prefix") -> pd.DataFrame:
    """
    Load multiple CSV files with a specific prefix from a fixed folder
    into a single pandas DataFrame, with memory/time tracking, filename column,
    and bus_id column based on filename (B183 -> 183, B208 -> 208).

    Args:
        path (str): Path to folder containing CSV files.
        prefix (str): Prefix of the filenames to match.

    Returns:
        pd.DataFrame: Concatenated DataFrame of all matched CSV files.
    """
    start_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)  # MB

    # File matching pattern
    file_pattern = os.path.join(path, f"{prefix}*.csv")
    csv_files = glob.glob(file_pattern)

    if not csv_files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

    # Load and tag each file with filename and bus_id
    df_list = []
    for file in csv_files:
        temp_df = pd.read_csv(file, engine='pyarrow')
        filename = os.path.splitext(os.path.basename(file))[0]  # Extract filename
        temp_df["source_file"] = filename
        
        # Assign bus_id based on filename
        # Metadata has this information
        #if "B183" in filename:
        #    temp_df["bus_id"] = 183
        #elif "B208" in filename:
        #    temp_df["bus_id"] = 208
        #else:
        #    temp_df["bus_id"] = None  # or np.nan

        df_list.append(temp_df)

    combined_df = pd.concat(df_list, ignore_index=True)

    mem_after = process.memory_info().rss / (1024 ** 2)  # MB
    elapsed_time = time.time() - start_time

    print(f"✅ Loaded {len(csv_files)} files into DataFrame")
    print(f"⏱ Load time: {elapsed_time:.2f} seconds")
    print(f"💾 Memory before: {mem_before:.2f} MB")
    print(f"💾 Memory after:  {mem_after:.2f} MB")
    print(f"📊 DataFrame shape: {combined_df.shape}")

    """
    Way to use the function

    data_path = "drive/folder/data"
    df = loadCSVToDataFrame(data_path, "prefix*")

    """

    return combined_df
