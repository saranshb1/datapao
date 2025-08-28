

import pandas as pd
import numpy as np

def reduceMemoryUsage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Initial size: {start_mem:.2f} MB")
    
    # Store column info
    col_info = []

    for col in df.columns:
        col_type = df[col].dtype
        c_min, c_max = None, None  # default if not numeric

        if col_type != object and not isinstance(col_type, pd.CategoricalDtype):
            c_min, c_max = df[col].min(), df[col].max()
            
            if str(col_type).startswith("int"):
                if c_min >= 0:
                    if c_max < 255:
                        df[col] = df[col].astype("uint8")
                    elif c_max < 65535:
                        df[col] = df[col].astype("uint16")
                    elif c_max < 4294967295:
                        df[col] = df[col].astype("uint32")
                    else:
                        df[col] = df[col].astype("uint64")
                else:
                    if np.iinfo("int8").min <= c_min <= np.iinfo("int8").max:
                        df[col] = df[col].astype("int8")
                    elif np.iinfo("int16").min <= c_min <= np.iinfo("int16").max:
                        df[col] = df[col].astype("int16")
                    elif np.iinfo("int32").min <= c_min <= np.iinfo("int32").max:
                        df[col] = df[col].astype("int32")
                    else:
                        df[col] = df[col].astype("int64")
                        
            elif str(col_type).startswith("float"):
                df[col] = pd.to_numeric(df[col], downcast="float")
                c_min, c_max = df[col].min(), df[col].max()  # recalc after downcast
                
        else:
            df[col] = df[col].astype("category")
        
        # Append column info
        col_info.append({
            "column": col,
            "dtype": df[col].dtype,
            "min": c_min,
            "max": c_max
        })
            
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Reduced size: {end_mem:.2f} MB ({100 * (start_mem - end_mem)/start_mem:.1f}% reduction)")

    # Convert collected info into DataFrame
    col_summary = pd.DataFrame(col_info)

    """

    Way to use the function

    df, df_summary = reduceMemoryUsage(df)

    """

    return df, col_summary
