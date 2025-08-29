

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plotColumnWithMissing(df: pd.DataFrame, column: str, 
                             title: str = None, save_path: str = None, 
                             figsize=(14, 5), color="blue", missing_color="red"):
    """
    Plot a DataFrame column over its index, highlighting missing values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column to plot
    column : str
        Name of the column to plot
    title : str, optional
        Title of the plot (default: "<column> with Missing Points")
    save_path : str, optional
        If provided, saves the plot to this file path
    figsize : tuple, default=(14, 5)
        Figure size
    color : str, default="blue"
        Line color for non-missing values
    missing_color : str, default="red"
        Marker color for missing values
    """
    plt.figure(figsize=figsize)
    
    # Plot main series
    plt.plot(df.index, df[column], color=color, alpha=0.4, label="Sensor Value")
    
    # Plot missing values
    missing_mask = df[column].isna()
    if missing_mask.any():
        plt.scatter(df.index[missing_mask],
                    [0] * missing_mask.sum(),
                    color=missing_color, marker="x", label="Missing", s=10)
    
    # Title
    if title is None:
        title = f"{column} with Missing Points"
    plt.title(title)
    plt.legend()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plotNumericCorrelations(df: pd.DataFrame, save_path: str = None, corr_threshold: float = 0.7):
    """
    Plot correlation heatmap for numerical features and list highly correlated pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features.
    corr_threshold : float, default=0.7
        Threshold above which features are considered correlated.

    Returns
    -------
    correlated_pairs : list of tuples
        List of correlated feature pairs with correlation above threshold.
    """
    # Select only numeric columns (all int/float types)
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Compute correlation matrix
    corr = numeric_df.corr()

    # --- Plot heatmap ---
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Correlation coefficient")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Numerical Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # --- Extract correlated features ---
    correlated_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > corr_threshold:
                correlated_pairs.append((
                    corr.columns[i],
                    corr.columns[j],
                    corr.iloc[i, j]
                ))
    
    return correlated_pairs

