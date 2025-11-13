# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 15:59:04 2025

@author: lm10_
"""

# ==========================================================
# =================== UTILITY FUNCTIONS ====================
# ==========================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# ----------------------------------------------------------
#  Separate features by type
# ----------------------------------------------------------

def _identify_feature_types(df, target_col):
    """Identify numerical, categorical, and sin/cos columns."""
    sin_cos_cols = [col for col in df.columns if 'sin' in col or 'cos' in col]
    cat_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns
                if col != target_col]
    num_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                if col not in sin_cos_cols and col != target_col]
    return num_cols, sin_cos_cols, cat_cols


# ----------------------------------------------------------
#  Preprocessing pipeline (fit/transform)
# ----------------------------------------------------------

def fit_transform_train(df_train, target_col, quantum=False):
    """
    Fit scalers/encoders on training data and transform it (RNN version).
    Keeps sin/cos untouched, normalizes numeric columns, one-hot encodes categorical columns.
    """
    num_cols, sin_cos_cols, cat_cols = _identify_feature_types(df_train, target_col)

    # Separate X and y
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    # Initialize transformers
    scaler = MinMaxScaler()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Scale numeric features
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_train[num_cols]),
        columns=num_cols, index=X_train.index
    )

    # Encode categorical
    X_cat = pd.DataFrame(
        encoder.fit_transform(X_train[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_train.index
    )

    # Concatenate all parts
    X_processed = pd.concat([X_scaled, X_train[sin_cos_cols], X_cat], axis=1)
    
    # If quantum = True, multiply continuous features by 2π
    if quantum:
        for col in num_cols:
            X_processed[col] = 2 * np.pi * X_processed[col]
            # sin/cos already in [-1, 1], we can leave them as is
            # categorical remain binary

    return X_processed, y_train, scaler, encoder


def transform_test(df_test, target_col, scaler, encoder, quantum=False):
    """Transform test data using pre-fitted scaler and encoder (RNN version)."""
    num_cols, sin_cos_cols, cat_cols = _identify_feature_types(df_test, target_col)
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    X_scaled = pd.DataFrame(
        scaler.transform(X_test[num_cols]),
        columns=num_cols, index=X_test.index
    )

    X_cat = pd.DataFrame(
        encoder.transform(X_test[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_test.index
    )

    X_processed = pd.concat([X_scaled, X_test[sin_cos_cols], X_cat], axis=1)
    
    # If quantum = True, multiply continuous features by 2π
    if quantum:
        for col in num_cols:
            X_processed[col] = 2 * np.pi * X_processed[col]
            # sin/cos already in [-1, 1], we can leave them as is
            # categorical remain binary
    
    return X_processed, y_test


# ----------------------------------------------------------
#  Window creation
# ----------------------------------------------------------

def create_windows_rnn(X_df, y_df, window_size):
    """
    Create 3D arrays for RNN input:
    X shape -> (num_samples, window_size, num_features)
    y shape -> (num_samples, 1)
    """
    X_rnn, y_rnn = [], []
    for i in range(len(X_df) - window_size):
        X_rnn.append(X_df.iloc[i:i+window_size].values)
        y_rnn.append(y_df.iloc[i + window_size])
    return np.array(X_rnn), np.array(y_rnn).reshape(-1, 1)


def create_windows_dnn(X_df, y_df, window_size,
                       use_moving_stats=True, aggregate=False):
    """
    Create flattened windows for DNN input using selected features.

    Parameters
    ----------
    X_df : pd.DataFrame
        Full feature dataframe.
    y_df : pd.Series
        Target series (aligned with X_df).
    window_size : int
        Number of timesteps per window.
    features : list[str]
        List of column names from X_df to include as features.
    use_moving_stats : bool, default=True
        Whether to include moving average and volatility per feature window.
    aggregate : bool, default=False
        If True, average each feature over the window (aggregate features instead of flattening).

    Returns
    -------
    X : np.ndarray, shape (samples, window_size * n_features + extras)
    y : np.ndarray, shape (samples, 1)
    """
    X_list, y_list = [], []
    
    features, sin_cos_cols, _ = _identify_feature_types(X_df, None)

    for i in range(len(X_df) - window_size):
        window = X_df[features].iloc[i:i+window_size].values

        if aggregate:
            # Aggregate across the window (mean per feature)
            feat_summary = np.mean(window, axis=0)
        else:
            # Flatten all time steps (sequence unfolded)
            feat_summary = window.flatten()

        # Optionally add moving statistics
        extras = []
        if use_moving_stats:
            moving_avg = np.mean(window)
            volatility = np.std(window)
            extras.extend([moving_avg, volatility])
            
        time_features = X_df[sin_cos_cols].iloc[i + window_size - 1].values

        X_list.append(np.hstack([feat_summary, extras, time_features]))
        y_list.append(y_df.iloc[i + window_size])

    return np.array(X_list), np.array(y_list).reshape(-1, 1)

