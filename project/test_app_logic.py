import pandas as pd
import joblib

def compute_features(df):
    df['OrderPerDay'] = df['OrderCount'] / (df['DaySinceLastOrder'] + 1)
    df['ComplainPerTenure'] = df['Complain'] / (df['Tenure'] + 1)
    df['IsNewCustomer'] = (df['Tenure'] <= 2).astype(int)
    return df

pipeline = joblib.load("churn_pipeline.pkl")
sample = pd.read_csv("sample_customers.csv")
print("CSV Columns:", sample.columns.tolist())

# Test single prediction exactly as app.py does
single_data = pd.DataFrame({
    'Tenure': [5],
    'DaySinceLastOrder': [10],
    'OrderCount': [2],
    'CashbackAmount': [150.0],
    'OrderAmountHikeFromlastYear': [12.0],
    'Complain': [1],
    'SatisfactionScore': [3],
    'WarehouseToHome': [15.0],
    'MaritalStatus': ["Single"],
    'PreferredPaymentMode': ["Credit Card"],
    'CityTier': [1]
})
single_data = compute_features(single_data)
prob = pipeline.predict_proba(single_data)[0, 1]
print("Single Prediction Risk %:", round(prob * 100, 1))

# Test CSV batch prediction
batch_data = sample.copy()
batch_data = compute_features(batch_data)
probs = pipeline.predict_proba(batch_data)[:, 1]
print("Batch predictions length:", len(probs))
print("App logic is successful!")
