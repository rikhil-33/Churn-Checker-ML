# Churn Check App 📉

A minimal, light-mode web app that predicts whether an e-commerce customer is likely to churn, powered by a trained Machine Learning pipeline and Streamlit.

## Features

- **Single Customer Risk Prediction**: Input customer tenure, order frequency, complaints, cashback, and satisfaction score to get an instant risk assessment.
- **Batch CSV Analysis**: Upload a customer dataset CSV to evaluate churn risk across cohorts.
- **Model Performance Metrics**: View model accuracy, ROC-AUC score, and confusion matrix statistics.

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rikhil-33/Churn-Checker-ML.git
   cd Churn-Checker-ML
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Navigate to the project directory**:
   ```bash
   cd project
   ```

4. **(Optional) Re-train the model**:
   ```bash
   python train_model.py
   ```

5. **Launch the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
