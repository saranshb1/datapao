

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def condenseMinMaxMean(df: pd.DataFrame, min_col: str, mean_col: str, max_col: str,
                          method: str = "weighted", weights: tuple = (0.2, 0.6, 0.2), 
                          lambda_: float = 0.1, new_col: str = "condensed_feature"):
    """
    Condense min, mean, max into one variable using different strategies.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    min_col, mean_col, max_col : str
        Column names for min, mean, and max.
    method : str, default "weighted"
        Method to condense features:
        - "weighted" → weighted average of (min, mean, max).
        - "range_mean" → mean + lambda * (max - min).
        - "pca" → first principal component of [min, mean, max].
    weights : tuple of length 3, default (0.2, 0.6, 0.2)
        Weights for (min, mean, max) when method="weighted".
    lambda_ : float, default 0.1
        Weight for variability (max - min) when method="range_mean".
    new_col : str, default "condensed_feature"
        Name of the new column.
    
    Returns
    -------
    - If method in {"weighted", "range_mean"}:
        pd.DataFrame
    - If method == "pca":
        (pd.DataFrame, PCA object)
    """
    
    df_copy = df.copy()
    
    if method == "weighted":
        a, b, c = weights
        df_copy[new_col] = a * df_copy[min_col] + b * df_copy[mean_col] + c * df_copy[max_col]
        return df_copy
    
    elif method == "range_mean":
        df_copy[new_col] = df_copy[mean_col] + lambda_ * (df_copy[max_col] - df_copy[min_col])
        return df_copy
    
    elif method == "pca":
        X = df_copy[[min_col, mean_col, max_col]].values
        pca = PCA(n_components=1)
        df_copy[new_col] = pca.fit_transform(X)
        return df_copy, pca
    
    else:
        raise ValueError("method must be 'weighted', 'range_mean', or 'pca'")


def condensePassengerFeatures(df: pd.Dataframe, feature_prefix: str, drop_original: bool = True, add_spread: bool = True):
    """
    Condense min, mean, max passenger features into interpretable condensed features.
    Keeps central tendency (mean), variability (range), and optional normalized spread.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns with suffixes '_min', '_mean', '_max'
    feature_prefix : str
        The prefix of the feature (e.g. 'passengers', 'speed')
    drop_original : bool, default=True
        If True, drop the original min/mean/max columns
    add_spread : bool, default=True
        If True, add normalized spread = (max - min) / mean

    Returns
    -------
    pd.DataFrame
        Updated dataframe with new condensed features
    """
    min_col = f"{feature_prefix}_min"
    mean_col = f"{feature_prefix}_mean"
    max_col = f"{feature_prefix}_max"

    # Safety check
    if not all(col in df.columns for col in [min_col, mean_col, max_col]):
        raise ValueError(f"One or more required columns ({min_col}, {mean_col}, {max_col}) are missing")

    # Create condensed features
    df[f"{feature_prefix}_central"] = df[mean_col]
    df[f"{feature_prefix}_range"] = df[max_col] - df[min_col]

    if add_spread:
        df[f"{feature_prefix}_spread"] = (
            (df[max_col] - df[min_col]) / df[mean_col].replace(0, np.nan)
        )

    # Drop originals if requested
    if drop_original:
        df = df.drop(columns=[min_col, mean_col, max_col])

    return df


def condenseFeaturesHighVariance(df: pd.DataFrame, feature_set: list, new_feature_name: str, 
                                 drop_original: bool = True, threshold: float = 0.99) -> pd.DataFrame:
    """
    Condense min, mean, max (or similar correlated features) into a single feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_set : list
        List of correlated features [min, mean, max]
    new_feature_name : str
        Name for the condensed feature
    drop_original : bool, default=True
        Whether to drop original features
    threshold : float, default=0.99
        Variance threshold for shortcut (if PCA explains variance above threshold, use mean)

    Returns
    -------
    pd.DataFrame
        DataFrame with new condensed feature
    """
    # Select features
    X = df[feature_set].dropna()

    # Fit PCA
    pca = PCA(n_components=1)
    pca.fit(X)

    explained_var = pca.explained_variance_ratio_[0]

    if explained_var >= threshold:
        # ✅ Strong collinearity → just take the mean
        df[new_feature_name] = df[feature_set].mean(axis=1)
        print(f"[INFO] Using MEAN instead of PCA (explained variance = {explained_var:.4f})")
    else:
        # ⚙️ Use PCA transformation
        df[new_feature_name] = pca.transform(df[feature_set])[:, 0]
        print(f"[INFO] Using PCA (explained variance = {explained_var:.4f})")
        print(f"[DEBUG] PCA components: {pca.components_}")

    # Drop original features if requested
    if drop_original:
        df = df.drop(columns=feature_set)

    return df

