import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBRegressor
import joblib
import shap
import time
import json
import os

def train_model(country_code):
    """
    Train XGBoost and LightGBM models to predict view velocity, analyze feature importance using SHAP.

    Args:
        country_code (str): Country code (e.g., 'US', 'KR').
    """
    # Ensure output directory exists
    output_dir = f'{country_code}_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"⏳ Starting training for country {country_code}...")

    # Load data
    start_time = time.time()
    df = pd.read_csv(f'{country_code}_processed_youtube_trending.csv')
    print(f" - Loaded data: {time.time() - start_time:.2f} seconds")

    # Select features and target
    features = [
        'view_count', 'likes', 'dislikes', 'comment_count',
        'engagement_rate', 'like_dislike_ratio', 'comment_view_ratio',
        'dislikes_per_comment', 'days_since_publication', 'likes_per_day',
        'comments_per_day', 'title_length', 'title_sentiment'
    ]
    target = 'view_velocity'

    # Prepare dataset
    X = df[features]
    y = df[target]

    # Handle missing values and infinities
    full_data = X.copy()
    full_data['target'] = y
    full_data = full_data.replace([np.inf, -np.inf], np.nan).dropna()

    # Split back
    X_clean = full_data[features]
    y_clean = full_data['target']

    # Save preprocessed data
    preprocessed_data = X_clean.copy()
    preprocessed_data[target] = y_clean
    additional_columns = ['video_id', 'tags', 'title', 'publishedAt', 'trending_date', 'categoryId']
    for col in additional_columns:
        if col in df.columns:
            preprocessed_data[col] = df.loc[preprocessed_data.index, col]
    preprocessed_data.to_csv(f'{country_code}_preprocessed_youtube_trending_data.csv', index=False)
    print(f"✅ Saved preprocessed file: {country_code}_preprocessed_youtube_trending_data.csv")
    print(f" - Processing time: {time.time() - start_time:.2f} seconds")

    # Compute sample weights to handle imbalanced data
    category_counts = df['categoryId'].value_counts()
    weights = df['categoryId'].map(lambda x: 1 / category_counts[x] if x in category_counts else 1)

    # Split train/test with weights
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X_clean, y_clean, weights, test_size=0.2, random_state=42
    )

    # Print shapes
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")

    # Check for anomalies
    print("Min and Max of features in X_train:")
    print(X_train.min())
    print(X_train.max())
    print("Min and Max of features in X_test:")
    print(X_test.min())
    print(X_test.max())

    # -------------------------- XGBoost --------------------------
    print("⏳ Training XGBoost...")
    start_time = time.time()
    param_grid_xgb = {
        'max_depth': [4, 6],
        'learning_rate': [0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8]
    }
    xgb_model = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        random_state=42
    )
    grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search_xgb.fit(X_train, y_train, sample_weight=weights_train)
    print(f"Best parameters for XGBoost ({country_code}): {grid_search_xgb.best_params_}")
    model_xgb = grid_search_xgb.best_estimator_

    # Predict
    y_pred_xgb = model_xgb.predict(X_test)

    # Evaluate
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mse_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)

    print(f"XGBoost Performance ({country_code}):")
    print(f"MAE: {mae_xgb:.6f}")
    print(f"MSE: {mse_xgb:.6f}")
    print(f"RMSE: {rmse_xgb:.6f}")
    print(f"R2 Score: {r2_xgb:.6f}")
    print(f" - XGBoost training time: {time.time() - start_time:.2f} seconds")

    # Plot predicted vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_xgb, alpha=0.5, label='XGBoost')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual View Velocity')
    plt.ylabel('Predicted View Velocity')
    plt.title(f'XGBoost Predicted vs Actual ({country_code})')
    plt.legend()
    plt.savefig(f'{output_dir}/{country_code}_xgb_pred_vs_actual.png')
    plt.close()

    # Feature importance
    print("Debug: Feature importance (XGBoost):")
    importance_xgb = model_xgb.get_booster().get_score(importance_type='weight')
    print(importance_xgb)
    if importance_xgb:
        xgb.plot_importance(model_xgb, max_num_features=10)
        plt.title(f'XGBoost Feature Importance ({country_code})')
        plt.savefig(f'{output_dir}/{country_code}_xgb_feature_importance.png')
        plt.close()

    # -------------------------- LightGBM --------------------------
    print("⏳ Training LightGBM...")
    start_time = time.time()
    param_grid_lgb = {
        'num_leaves': [20, 31],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'bagging_fraction': [0.8]
    }
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        boosting_type='gbdt',
        verbose=-1
    )
    grid_search_lgb = GridSearchCV(lgb_model, param_grid_lgb, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search_lgb.fit(X_train, y_train, sample_weight=weights_train)
    print(f"Best parameters for LightGBM ({country_code}): {grid_search_lgb.best_params_}")
    model_lgb = grid_search_lgb.best_estimator_

    # Predict
    y_pred_lgb = model_lgb.predict(X_test)

    # Evaluate
    mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
    mse_lgb = mean_squared_error(y_test, y_pred_lgb)
    rmse_lgb = np.sqrt(mse_lgb)
    r2_lgb = r2_score(y_test, y_pred_lgb)

    print(f"\nLightGBM Performance ({country_code}):")
    print(f"MAE: {mae_lgb:.6f}")
    print(f"MSE: {mse_lgb:.6f}")
    print(f"RMSE: {rmse_lgb:.6f}")
    print(f"R2 Score: {r2_lgb:.6f}")
    print(f" - LightGBM training time: {time.time() - start_time:.2f} seconds")

    # Plot predicted vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_lgb, alpha=0.5, label='LightGBM')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual View Velocity')
    plt.ylabel(' Predicted View Velocity')
    plt.title(f'LightGBM Predicted vs Actual ({country_code})')
    plt.legend()
    plt.savefig(f'{output_dir}/{country_code}_lgb_pred_vs_actual.png')
    plt.close()

    # Feature importance
    print("Debug: Feature importance (LightGBM):")
    importance_lgb = model_lgb.feature_importances_
    print(importance_lgb)
    if importance_lgb.sum() > 0:
        lgb.plot_importance(model_lgb, max_num_features=10)
        plt.title(f'LightGBM Feature Importance ({country_code})')
        plt.savefig(f'{output_dir}/{country_code}_lgb_feature_importance.png')
        plt.close()

    # Compare feature importance
    importance_xgb_series = pd.Series(importance_xgb, name='XGBoost')
    importance_lgb_series = pd.Series(importance_lgb, index=features, name='LightGBM')
    importance_df = pd.concat([importance_xgb_series, importance_lgb_series], axis=1).fillna(0)
    plt.figure(figsize=(12, 6))
    importance_df.plot(kind='bar', title=f'Feature Importance Comparison ({country_code})')
    plt.savefig(f'{output_dir}/{country_code}_feature_importance_comparison.png')
    plt.close()

    # Compare models
    results = pd.DataFrame({
        'Model': ['XGBoost', 'LightGBM'],
        'MAE': [mae_xgb, mae_lgb],
        'MSE': [mse_xgb, mse_lgb],
        'RMSE': [rmse_xgb, rmse_lgb],
        'R2 Score': [r2_xgb, r2_lgb]
    })
    print(f"\nModel Comparison ({country_code}):")
    print(results)

    # Select best model
    if abs(r2_xgb - r2_lgb) < 0.0001:
        best_model_name = "XGBoost"
        best_model = model_xgb
    elif r2_xgb > r2_lgb:
        best_model_name = "XGBoost"
        best_model = model_xgb
    else:
        best_model_name = "LightGBM"
        best_model = model_lgb

    print(f"\n✅ Best Model Selected for {country_code}: {best_model_name}")

    # Save best model info
    model_info = {"best_model": best_model_name}
    with open(f'{output_dir}/{country_code}_best_model.json', 'w') as f:
        json.dump(model_info, f)

    # Clean X_test for SHAP
    print("Debug: Checking X_test before SHAP computation...")
    if X_test.isna().any().any():
        print("Warning: X_test contains NaN values. Removing...")
        X_test = X_test.dropna()
    if np.isinf(X_test.to_numpy()).any():
        print("Warning: X_test contains infinite values. Replacing...")
        X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()

    # Save model and SHAP values
    print("⏳ Computing SHAP values...")
    start_time = time.time()
    if best_model_name == "XGBoost":
        joblib.dump(model_xgb, f'{output_dir}/{country_code}_model_xgb.joblib')
        explainer = shap.Explainer(model_xgb, X_test)
        shap_values = explainer(X_test)
        joblib.dump(shap_values, f'{output_dir}/{country_code}_shap_values_xgb.joblib')
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(f'{output_dir}/{country_code}_shap_summary.png')
        plt.close()
    else:
        joblib.dump(model_lgb, f'{output_dir}/{country_code}_model_lgb.joblib')
        explainer = shap.Explainer(model_lgb, X_test)
        shap_values = explainer(X_test, check_additivity=False)
        joblib.dump(shap_values, f'{output_dir}/{country_code}_shap_values_lgb.joblib')
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(f'{output_dir}/{country_code}_shap_summary.png')
        plt.close()
    print(f" - SHAP computation time: {time.time() - start_time:.2f} seconds")

    # SHAP analysis per category
    print("⏳ Analyzing SHAP per category...")
    start_time = time.time()
    for category_id in df['categoryId'].unique():
        category_data = X_test[df.loc[X_test.index, 'categoryId'] == category_id]
        if len(category_data) > 0:
            shap_values_cat = explainer(category_data, check_additivity=False)
            shap.summary_plot(shap_values_cat, category_data, plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance for Category {category_id} ({country_code})")
            plt.savefig(f'{output_dir}/{country_code}_shap_category_{category_id}.png')
            plt.close()
    print(f" - SHAP per category analysis time: {time.time() - start_time:.2f} seconds")

    # Save X_test for recommendation
    X_test.to_csv(f'{output_dir}/{country_code}_X_test.csv', index=True)

if __name__ == "__main__":
    countries = ['US', 'IN', 'BR', 'GB', 'KR']
    for country in countries:
        train_model(country)