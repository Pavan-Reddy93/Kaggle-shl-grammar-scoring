import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_cv_pipeline(features_df, features_cols):
    """
    Trains a 5-fold cross-validated Random Forest Regressor and prints local OOF metrics.
    """
    oof_predictions = np.zeros(len(features_df))
    models = []
    scalers = []
    
    for fold in range(5):
        train_idx = features_df[features_df['fold'] != fold].index
        val_idx = features_df[features_df['fold'] == fold].index

        X_train, y_train = features_df.loc[train_idx, features_cols], features_df.loc[train_idx, 'label']
        X_val, y_val = features_df.loc[val_idx, features_cols], features_df.loc[val_idx, 'label']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        oof_predictions[val_idx] = model.predict(X_val_scaled)
        
        models.append(model)
        scalers.append(scaler)
        
    # Overall local metrics
    overall_rmse = np.sqrt(mean_squared_error(features_df['label'], oof_predictions))
    overall_pearson, _ = pearsonr(features_df['label'], oof_predictions)
    print(f"Local OOF RF RMSE: {overall_rmse:.4f}")
    print(f"Local OOF RF Pearson Correlation: {overall_pearson:.4f}")
    
    return models, scalers