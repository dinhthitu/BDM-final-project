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

def train_model(country_code):
    print(f"⏳ Bắt đầu huấn luyện cho quốc gia {country_code}...")

    # Load dữ liệu
    start_time = time.time()
    df = pd.read_csv(f'{country_code}_processed_youtube_trending.csv')
    print(f" - Đã load dữ liệu: {time.time() - start_time:.2f} giây")

    # Chọn features và target
    features = [
        'view_count', 'likes', 'dislikes', 'comment_count',
        'engagement_rate', 'like_dislike_ratio', 'comment_view_ratio',
        'dislikes_per_comment', 'days_since_publication', 'likes_per_day',
        'comments_per_day', 'title_length', 'title_sentiment'
    ]
    target = 'view_velocity'

    # Lấy tập dữ liệu chứa đầy đủ cả features và target
    X = df[features]
    y = df[target]

    # Gộp lại để xử lý missing values cho toàn bộ
    full_data = X.copy()
    full_data['target'] = y
    full_data = full_data.replace([np.inf, -np.inf], np.nan).dropna()

    # Tách lại
    X_clean = full_data[features]
    y_clean = full_data['target']

    # Lưu lại file đã được tiền xử lý
    preprocessed_data = X_clean.copy()
    preprocessed_data[target] = y_clean
    additional_columns = ['video_id', 'tags', 'title', 'publishedAt', 'trending_date', 'categoryId']
    for col in additional_columns:
        if col in df.columns:
            preprocessed_data[col] = df.loc[preprocessed_data.index, col]
    preprocessed_data.to_csv(f'{country_code}_preprocessed_youtube_trending_data.csv', index=False)
    print(f"✅ Đã lưu file tiền xử lý: f'{country_code}_preprocessed_youtube_trending_data.csv'")
    print(f" - Thời gian xử lý: {time.time() - start_time:.2f} giây")

    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

    # Kiểm tra shape của X_train và X_test
    print(f"Shape của X_train: {X_train.shape}")
    print(f"Shape của X_test: {X_test.shape}")

    # Kiểm tra giá trị bất thường
    print("Min và Max của từng đặc trưng trong X_train:")
    print(X_train.min())
    print(X_train.max())
    print("Min và Max của từng đặc trưng trong X_test:")
    print(X_test.min())
    print(X_test.max())

    # -------------------------- XGBoost --------------------------
    print("⏳ Huấn luyện XGBoost...")
    start_time = time.time()
    
    # Tinh chỉnh tham số bằng GridSearchCV (giảm phạm vi để tăng tốc)
    param_grid_xgb = {
        'max_depth': [4, 6],  # Giảm số lựa chọn
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
    grid_search_xgb.fit(X_train, y_train)
    print(f"Best parameters for XGBoost ({country_code}): {grid_search_xgb.best_params_}")
    model_xgb = grid_search_xgb.best_estimator_

    # Dự đoán
    y_pred_xgb = model_xgb.predict(X_test)

    # Đánh giá
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mse_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)

    print(f"XGBoost Performance ({country_code}):")
    print(f"MAE: {mae_xgb:.6f}")
    print(f"MSE: {mse_xgb:.6f}")
    print(f"RMSE: {rmse_xgb:.6f}")
    print(f"R2 Score: {r2_xgb:.6f}")
    print(f" - Thời gian huấn luyện XGBoost: {time.time() - start_time:.2f} giây")

    # Feature importance
    print("Debug: Tầm quan trọng của các đặc trưng (XGBoost):")
    importance = model_xgb.get_booster().get_score(importance_type='weight')
    print(importance)
    if importance:
        xgb.plot_importance(model_xgb, max_num_features=10)
        plt.title(f'XGBoost Feature Importance ({country_code})')
        plt.savefig(f'{country_code}_output/{country_code}_xgb_feature_importance.png')  # Lưu trước
        plt.show()  # Hiển thị sau
        plt.close()
    else:
        print("Cảnh báo: Không có đặc trưng nào có tầm quan trọng (XGBoost).")

    # -------------------------- LightGBM --------------------------
    print("⏳ Huấn luyện LightGBM...")
    start_time = time.time()
    
    # Tinh chỉnh tham số bằng GridSearchCV (giảm phạm vi để tăng tốc)
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
    grid_search_lgb.fit(X_train, y_train)
    print(f"Best parameters for LightGBM ({country_code}): {grid_search_lgb.best_params_}")
    model_lgb = grid_search_lgb.best_estimator_

    # Dự đoán
    y_pred_lgb = model_lgb.predict(X_test)

    # Đánh giá
    mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
    mse_lgb = mean_squared_error(y_test, y_pred_lgb)
    rmse_lgb = np.sqrt(mse_lgb)
    r2_lgb = r2_score(y_test, y_pred_lgb)

    print(f"\nLightGBM Performance ({country_code}):")
    print(f"MAE: {mae_lgb:.6f}")
    print(f"MSE: {mse_lgb:.6f}")
    print(f"RMSE: {rmse_lgb:.6f}")
    print(f"R2 Score: {r2_lgb:.6f}")
    print(f" - Thời gian huấn luyện LightGBM: {time.time() - start_time:.2f} giây")

    # Feature importance
    print("Debug: Tầm quan trọng của các đặc trưng (LightGBM):")
    importance = model_lgb.feature_importances_
    print(importance)
    if importance.sum() > 0:
        lgb.plot_importance(model_lgb, max_num_features=10)
        plt.title(f'LightGBM Feature Importance ({country_code})')
        plt.savefig(f'{country_code}_output/{country_code}_lgb_feature_importance.png')  # Lưu trước
        plt.show()  # Hiển thị sau
        plt.close()
    else:
        print("Cảnh báo: Không có đặc trưng nào có tầm quan trọng (LightGBM).")

    # So sánh
    results = pd.DataFrame({
        'Model': ['XGBoost', 'LightGBM'],
        'MAE': [mae_xgb, mae_lgb],
        'MSE': [mse_xgb, mse_lgb],
        'RMSE': [rmse_xgb, rmse_lgb],
        'R2 Score': [r2_xgb, r2_lgb]
    })
    print(f"\nModel Comparison ({country_code}):")
    print(results)

    # Lựa chọn best model
    if abs(r2_xgb - r2_lgb) < 0.0001:  # Nếu R2 Score gần nhau, chọn XGBoost để tránh lỗi SHAP
        best_model_name = "XGBoost"
        best_model = model_xgb
    elif r2_xgb > r2_lgb:
        best_model_name = "XGBoost"
        best_model = model_xgb
    else:
        best_model_name = "LightGBM"
        best_model = model_lgb

    print(f"\n✅ Best Model Selected for {country_code}: {best_model_name}")

    # Lưu best model name vào file JSON
    model_info = {"best_model": best_model_name}
    with open(f'{country_code}_output/{country_code}_best_model.json', 'w') as f:
        json.dump(model_info, f)

    # Kiểm tra và làm sạch X_test trước khi tính SHAP
    print("Debug: Kiểm tra dữ liệu X_test trước khi tính SHAP...")
    if X_test.isna().any().any():
        print("Cảnh báo: X_test chứa giá trị NaN. Đang loại bỏ...")
        X_test = X_test.dropna()
    if np.isinf(X_test.to_numpy()).any():
        print("Cảnh báo: X_test chứa giá trị vô cực (inf). Đang thay thế...")
        X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()

    # Đảm bảo X_test có cùng shape với dữ liệu huấn luyện
    if X_test.shape[1] != len(features):
        raise ValueError(f"Shape của X_test ({X_test.shape[1]}) không khớp với số đặc trưng ({len(features)}).")

    # Lưu model và SHAP values
    print("⏳ Tính toán SHAP values...")
    start_time = time.time()
    if best_model_name == "XGBoost":
        joblib.dump(model_xgb, f'{country_code}_output/{country_code}_model_xgb.joblib')
        explainer = shap.Explainer(model_xgb, X_test)
        shap_values = explainer(X_test)
        joblib.dump(shap_values, f'{country_code}_output/{country_code}_shap_values_xgb.joblib')
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(f'{country_code}_output/{country_code}_shap_summary.png')
        plt.close()
    else:
        joblib.dump(model_lgb, f'{country_code}_output/{country_code}_model_lgb.joblib')
        explainer = shap.Explainer(model_lgb, X_test)
        shap_values = explainer(X_test, check_additivity=False)  # Tắt kiểm tra additivity cho LightGBM
        joblib.dump(shap_values, f'{country_code}_output/{country_code}_shap_values_lgb.joblib')
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(f'{country_code}_output/{country_code}_shap_summary.png')
        plt.close()
    print(f" - Thời gian tính SHAP: {time.time() - start_time:.2f} giây")

    # Phân tích SHAP theo từng danh mục
    print("⏳ Phân tích SHAP theo danh mục...")
    start_time = time.time()
    for category_id in df['categoryId'].unique():
        category_data = X_test[df.loc[X_test.index, 'categoryId'] == category_id]
        if len(category_data) > 0:
            shap_values_cat = explainer(category_data, check_additivity=False)
            shap.summary_plot(shap_values_cat, category_data, plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance for Category {category_id} ({country_code})")
            plt.savefig(f'{country_code}_output/{country_code}_shap_category_{category_id}.png')
            plt.close()
    print(f" - Thời gian phân tích SHAP theo danh mục: {time.time() - start_time:.2f} giây")

    # Lưu X_test để sử dụng trong Recommendation.py
    X_test.to_csv(f'{country_code}_output/{country_code}_X_test.csv', index=True)

if __name__ == "__main__":
    countries = ['US', 'IN', 'BR', 'GB', 'KR']
    for country in countries:
        train_model(country)