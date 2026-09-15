import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Data generation & target transformation
np.random.seed(42)
n_samples = 1000

data = {
    'customer_id': [f'CUST_{i:04d}' for i in range(n_samples)],
    'first_purchase_date': pd.date_range(start='2026-01-01', periods=n_samples, freq='h'),
    'category_first_item': np.random.choice(['Dresses', 'Sneakers', 'Accessories', 'T-Shirts'], n_samples),
    'first_checkout_amount': np.random.exponential(scale=150, size=n_samples) + 20,
    'pages_viewed_week1': np.random.poisson(lam=12, size=n_samples),
    'device_type': np.random.choice(['iOS', 'Android', 'Web'], n_samples, p=[0.35, 0.45, 0.20]),
    'discount_used_week1': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
}
df = pd.DataFrame(data)

base_ltv = df['first_checkout_amount'] * np.random.uniform(1.2, 3.0, n_samples)
ios_premium = (df['device_type'] == 'iOS') & (df['discount_used_week1'] == 0)
high_intent = (df['pages_viewed_week1'] > 15) & (df['first_checkout_amount'] > 200)
df['target_ltv_year'] = (base_ltv + (ios_premium * 800) + (high_intent * 1200)).round(2)

# 2. Feature engineering (Custom psychology-driven features)
df['feature_engagement_index'] = df['pages_viewed_week1'] * (df['first_checkout_amount'] / 100.0)
df['feature_ios_premium_wallet'] = ((df['device_type'] == 'iOS') & (df['discount_used_week1'] == 0)).astype(int)
df['feature_intent_score'] = (df['first_checkout_amount'] * df['pages_viewed_week1']) / (df['discount_used_week1'] + 1)

# 3. Time-based split & log-transformation
df = df.sort_values('first_purchase_date').reset_index(drop=True)
train_size = int(len(df) * 0.8)

features = ['first_checkout_amount', 'pages_viewed_week1', 'discount_used_week1', 
            'feature_engagement_index', 'feature_ios_premium_wallet', 'feature_intent_score',
            'category_first_item', 'device_type']

X = pd.get_dummies(df[features], drop_first=True)
y_log = np.log1p(df['target_ltv_year']) 

X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train_log, y_test_log = y_log.iloc[:train_size], y_log.iloc[train_size:]
y_test_original = df['target_ltv_year'].iloc[train_size:]

# 4. Training Gradient Boosting
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.07, max_depth=4, random_state=42)
model.fit(X_train, y_train_log)

# 5. Predictions & Evaluation
train_preds = model.predict(X_train)
test_preds_log = model.predict(X_test)
predictions = np.expm1(test_preds_log)

# Validation Metrics
train_mae = mean_absolute_error(np.expm1(y_train_log), np.expm1(train_preds))
test_mae = mean_absolute_error(y_test_original, predictions)
test_rmse = np.sqrt(mean_squared_error(y_test_original, predictions))
feat_imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

print(f"Train MAE: {train_mae:.2f}")
print(f"Test MAE: {test_mae:.2f} | Test RMSE: {test_rmse:.2f}")
print("\nTop Features:\n", feat_imp.head(3))

# 6. Model Serialization
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/ltv_model.pkl')
joblib.dump(X_train.columns.tolist(), 'models/model_features.pkl')
print("\n💾 Model and feature schema successfully saved to 'models/' folder!")
