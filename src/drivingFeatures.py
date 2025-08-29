

import pandas as pd

def reduceDrivingFeatures(df: pd.DataFrame, corr_threshold: float = 0.9) -> pd.DataFrame:
    """
    Reduce odometry/traction/status features by combining redundant ones and
    removing highly correlated columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with all features.
    corr_threshold : float, default=0.9
        Threshold for removing correlated features.
    
    Returns
    -------
    pd.DataFrame
        Reduced dataframe with fewer, more informative features.
    """
    
    # --- Domain-driven reduction ---
    # Combine wheel speeds into mean and differences
    df = df.copy()
    
    # --- Domain-driven reduction ---
    wheel_cols = [
        'odometry_wheelSpeed_fl', 'odometry_wheelSpeed_fr',
        'odometry_wheelSpeed_ml', 'odometry_wheelSpeed_mr',
        'odometry_wheelSpeed_rl', 'odometry_wheelSpeed_rr'
    ]
    
    # Fill NaNs in wheel speeds with 0 (or you can choose another strategy)
    df[wheel_cols] = df[wheel_cols].fillna(0)
    
    # Mean wheel speed
    df['wheelSpeed_mean'] = df[wheel_cols].mean(axis=1)
    
    # Left-right difference
    df['wheelSpeed_left_right_diff'] = (
        (df['odometry_wheelSpeed_fl'] + df['odometry_wheelSpeed_ml'] + df['odometry_wheelSpeed_rl']) / 3
        - (df['odometry_wheelSpeed_fr'] + df['odometry_wheelSpeed_mr'] + df['odometry_wheelSpeed_rr']) / 3
    )
    
    # Combine brake flags into a single categorical feature
    df['brake_status'] = (
        pd.to_numeric(df['status_haltBrakeIsActive'], errors='coerce').fillna(0).astype(int) +
        2 * pd.to_numeric(df['status_parkBrakeIsActive'], errors='coerce').fillna(0).astype(int)
        )
    # 0 = no brake, 1 = halt brake, 2 = park brake, 3 = both
    
    # Drop raw features already aggregated
    drop_cols = [
        'odometry_wheelSpeed_fl', 'odometry_wheelSpeed_fr',
        'odometry_wheelSpeed_ml', 'odometry_wheelSpeed_mr',
        'odometry_wheelSpeed_rl', 'odometry_wheelSpeed_rr',
        'status_haltBrakeIsActive', 'status_parkBrakeIsActive'
    ]
    df_reduced = df.drop(columns=drop_cols, errors="ignore")

    #These features are generalized versions of the metadata. We do not need them for prediction purposes.
    drop_generalized = ['drivenDistance', 'energyConsumption']
    
    df_reduced = df_reduced.drop(columns=drop_generalized, errors="ignore")
    
    # --- Statistical reduction ---
    # Remove highly correlated features
    #corr = df_reduced.corr().abs()
    #upper = corr.where(~pd.np.tril(pd.np.ones(corr.shape)).astype(bool))  # upper triangle
    #to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    
    #df_final = df_reduced.drop(columns=to_drop, errors="ignore")
    
    return df_reduced



