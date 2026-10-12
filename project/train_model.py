import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def main():
    print("Loading data...")
    df = pd.read_excel("E Commerce Dataset 2.xlsx", sheet_name="E Comm")
    
    # 1. Fix inconsistent labels
    df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace('Phone', 'Mobile Phone')
    df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({'CC': 'Credit Card', 'COD': 'Cash on Delivery'})
    df['PreferedOrderCat'] = df['PreferedOrderCat'].replace('Mobile', 'Mobile Phone')
    
    # 2. Cap extreme outliers (WarehouseToHome using IQR, etc.)
    for col in ['WarehouseToHome']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        
    # 3. Feature engineering
    df['OrderPerDay'] = df['OrderCount'] / (df['DaySinceLastOrder'] + 1)
    df['ComplainPerTenure'] = df['Complain'] / (df['Tenure'] + 1)
    df['IsNewCustomer'] = (df['Tenure'] <= 2).astype(int)
    
    # 4. Select features to keep
    features_to_keep = [
        'Tenure', 'DaySinceLastOrder', 'OrderCount', 'CashbackAmount', 
        'OrderAmountHikeFromlastYear', 'Complain', 'SatisfactionScore', 
        'WarehouseToHome', 'MaritalStatus', 'PreferredPaymentMode', 'CityTier',
        'OrderPerDay', 'ComplainPerTenure', 'IsNewCustomer'
    ]
    target = 'Churn'
    
    X = df[features_to_keep]
    y = df[target]
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Save a sample of the test set for batch upload demo
    sample_df = X_test.head(20).copy()
    sample_df.to_csv("sample_customers.csv", index=False)
    
    print("Building pipelines...")
    numeric_features = [
        'Tenure', 'DaySinceLastOrder', 'OrderCount', 'CashbackAmount', 
        'OrderAmountHikeFromlastYear', 'Complain', 'SatisfactionScore', 
        'WarehouseToHome', 'CityTier', 'OrderPerDay', 'ComplainPerTenure', 'IsNewCustomer'
    ]
    categorical_features = ['MaritalStatus', 'PreferredPaymentMode']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
        'XGBoost (HistGradient)': HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    }
    
    best_model_name = None
    best_f1 = -1
    best_pipeline = None
    all_metrics = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        
        all_metrics[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'ROC-AUC': roc
        }
        
        if name == 'XGBoost (HistGradient)':
            best_model_name = name
            best_pipeline = pipeline
            
    print(f"Best model selected: {best_model_name}")
    joblib.dump(best_pipeline, 'churn_pipeline.pkl')
    
    # Extract feature importances from Random Forest (since HistGradient doesn't have it natively)
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', models['Random Forest'])])
    rf_pipeline.fit(X_train, y_train)
    rf_model = rf_pipeline.named_steps['classifier']
    cat_encoder = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_cols = cat_encoder.get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_cols)
    importances = rf_model.feature_importances_
    
    # Keep top 10 features for the chart
    importance_dict = {f: float(imp) for f, imp in zip(feature_names, importances)}
    top_features = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)[:10])
    
    metrics_data = {
        'model_comparison': all_metrics,
        'best_model': best_model_name,
        'best_metrics': all_metrics[best_model_name],
        'top_features': top_features
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=4)
        
    print("Training complete. Models and metrics saved.")

if __name__ == "__main__":
    main()
