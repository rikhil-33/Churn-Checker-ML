import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="Churn Check", page_icon="📉", layout="centered")

# --- Custom CSS for clean UI ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .risk-low { color: #10b981; font-weight: bold; font-size: 24px; }
    .risk-medium { color: #f59e0b; font-weight: bold; font-size: 24px; }
    .risk-high { color: #ef4444; font-weight: bold; font-size: 24px; }
    .stButton button { width: 100%; border-radius: 8px; background-color: #5c6bc0; color: white; }
    .stButton button:hover { background-color: #3f51b5; color: white; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / 'churn_pipeline.pkl')

@st.cache_data
def load_metrics():
    with open(BASE_DIR / 'metrics.json', 'r') as f:
        return json.load(f)

try:
    pipeline = load_model()
    metrics = load_metrics()
except Exception as e:
    st.error(f"Error loading model or metrics. Please run train_model.py first. ({e})")
    st.stop()

def compute_features(df):
    df['OrderPerDay'] = df['OrderCount'] / (df['DaySinceLastOrder'] + 1)
    df['ComplainPerTenure'] = df['Complain'] / (df['Tenure'] + 1)
    df['IsNewCustomer'] = (df['Tenure'] <= 2).astype(int)
    return df

st.title("📉 Churn Check")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📂 Batch Upload", "ℹ️ About the Model"])

with tab1:
    st.markdown("### Will this customer leave?")
    st.markdown("<span style='color:#64748b;'>Enter the customer details below to check their churn risk.</span>", unsafe_allow_html=True)
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=5, help="How many months the customer has been with us")
            day_since_last_order = st.number_input("Days Since Last Order", min_value=0, max_value=100, value=10)
            order_count = st.number_input("Order Count", min_value=1, max_value=50, value=2)
            cashback = st.number_input("Cashback Amount ($)", min_value=0.0, max_value=500.0, value=150.0)
            order_hike = st.number_input("Order Amount Hike (%)", min_value=0.0, max_value=100.0, value=12.0, help="Increase from last year")
            complain = st.selectbox("Filed a Complain?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            
        with col2:
            satisfaction = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3)
            warehouse_to_home = st.number_input("Distance to Home (km)", min_value=0.0, max_value=200.0, value=15.0)
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            payment_mode = st.selectbox("Preferred Payment Mode", ["Debit Card", "Credit Card", "E wallet", "UPI", "Cash on Delivery"])
            city_tier = st.selectbox("City Tier", [1, 2, 3])
            
        submitted = st.form_submit_button("Check risk")
        
    if submitted:
        input_data = pd.DataFrame({
            'Tenure': [tenure],
            'DaySinceLastOrder': [day_since_last_order],
            'OrderCount': [order_count],
            'CashbackAmount': [cashback],
            'OrderAmountHikeFromlastYear': [order_hike],
            'Complain': [complain],
            'SatisfactionScore': [satisfaction],
            'WarehouseToHome': [warehouse_to_home],
            'MaritalStatus': [marital_status],
            'PreferredPaymentMode': [payment_mode],
            'CityTier': [city_tier]
        })
        input_data = compute_features(input_data)
        
        prob = pipeline.predict_proba(input_data)[0, 1]
        risk_pct = round(prob * 100, 1)
        
        st.markdown("---")
        # Gauge Chart
        if risk_pct < 30:
            color = "#10b981"
            level = "Low risk"
            css_class = "risk-low"
        elif risk_pct < 70:
            color = "#f59e0b"
            level = "Medium risk"
            css_class = "risk-medium"
        else:
            color = "#ef4444"
            level = "High risk"
            css_class = "risk-high"
            
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_pct,
            number = {'suffix': "%", 'font': {'color': "#2c3e50"}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#ecfdf5'},
                    {'range': [30, 70], 'color': '#fffbeb'},
                    {'range': [70, 100], 'color': '#fef2f2'}],
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"<div style='text-align: center;'><span class='{css_class}'>{level}</span></div><br>", unsafe_allow_html=True)
        
        # Reasons and Actions
        st.markdown("#### Key Drivers & Recommendations")
        reasons = []
        if tenure <= 2:
            reasons.append("- **New Customer:** Early in their journey, higher chance of leaving.")
        if complain == 1:
            reasons.append("- **Recent Complaint:** Dissatisfaction is a strong churn signal.")
        if satisfaction <= 2:
            reasons.append("- **Low Satisfaction:** Scored poorly on their last interaction.")
        if not reasons:
            reasons.append("- Customer profile is relatively stable.")
            
        for r in reasons:
            st.markdown(r)
            
        st.markdown("**Suggested Action:**")
        if complain == 1 or risk_pct >= 70:
            st.info("🎯 **Prioritize support follow-up** and offer a retention discount.")
        elif tenure <= 2:
            st.info("👋 **Send a welcome offer** or onboarding email to build loyalty.")
        else:
            st.success("✅ Continue standard engagement. No immediate intervention needed.")

with tab2:
    st.markdown("### Batch Upload")
    st.write("Upload a CSV file of multiple customers to score them all at once.")
    
    try:
        sample_df = pd.read_csv(BASE_DIR / "sample_customers.csv")
        csv = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download sample CSV",
            data=csv,
            file_name='sample_customers.csv',
            mime='text/csv',
        )
    except FileNotFoundError:
        st.warning("Sample CSV not found. Run train_model.py first.")
        
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # check columns
            req_cols = ['Tenure', 'DaySinceLastOrder', 'OrderCount', 'CashbackAmount', 
                        'OrderAmountHikeFromlastYear', 'Complain', 'SatisfactionScore', 
                        'WarehouseToHome', 'MaritalStatus', 'PreferredPaymentMode', 'CityTier']
            missing = [c for c in req_cols if c not in batch_df.columns]
            
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                score_df = batch_df.copy()
                score_df = compute_features(score_df)
                probs = pipeline.predict_proba(score_df)[:, 1]
                
                batch_df['Risk %'] = (probs * 100).round(1)
                batch_df['Risk level'] = pd.cut(batch_df['Risk %'], bins=[-1, 30, 70, 101], labels=['Low', 'Medium', 'High'])
                
                # Assume first column is identifier if it's named like id or customer, else just index
                first_col = batch_df.columns[0]
                display_cols = [first_col, 'Risk %', 'Risk level']
                display_df = batch_df[display_cols].sort_values(by='Risk %', ascending=False)
                
                filter_high = st.checkbox("Show 'High risk' only")
                if filter_high:
                    display_df = display_df[display_df['Risk level'] == 'High']
                    
                st.dataframe(display_df, use_container_width=True)
                
                out_csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download results", data=out_csv, file_name="scored_customers.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

with tab3:
    st.markdown("### About the Model")
    
    best_name = metrics['best_model']
    m = metrics['best_metrics']
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Recall", f"{m['Recall']*100:.1f}%")
    col2.metric("ROC-AUC", f"{m['ROC-AUC']:.2f}")
    col3.metric("Accuracy", f"{m['Accuracy']*100:.1f}%")
    
    st.markdown("""
    *Why Recall matters more than Accuracy:* Since only about 17% of customers churn, a model could simply predict "no one churns" and achieve 83% accuracy. **Recall** ensures we actually catch the customers who are leaving, which is the real goal for retention teams!
    """)
    
    st.markdown("#### Model Comparison")
    comp_df = pd.DataFrame(metrics['model_comparison']).T
    comp_df = (comp_df * 100).round(1).astype(str) + "%"
    # fix ROC format back to decimal
    for model in metrics['model_comparison']:
        comp_df.loc[model, 'ROC-AUC'] = f"{metrics['model_comparison'][model]['ROC-AUC']:.2f}"
    st.dataframe(comp_df, use_container_width=True)
    
    st.markdown("#### Top Features Driving Churn")
    feats = metrics['top_features']
    fig_feat = px.bar(
        x=list(feats.values()), y=list(feats.keys()), 
        orientation='h', 
        labels={'x': 'Importance', 'y': 'Feature'}
    )
    fig_feat.update_layout(yaxis={'categoryorder':'total ascending'}, height=350, margin=dict(l=0, r=0, t=0, b=0))
    fig_feat.update_traces(marker_color='#5c6bc0')
    st.plotly_chart(fig_feat, use_container_width=True)
    
    st.markdown("**Key Finding:** Customers with low tenure (new customers) and those who have filed complaints are historically the most likely to leave.")
