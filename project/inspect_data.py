import pandas as pd

excel_file = "E Commerce Dataset 2.xlsx"
xl = pd.ExcelFile(excel_file)
print("Sheets:", xl.sheet_names)

df = pd.read_excel(excel_file, sheet_name="E Comm")
print("\n--- Columns & Dtypes ---")
print(df.dtypes)
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Categorical Value Counts ---")
for col in df.select_dtypes(include=['object']).columns:
    print(f"\n{col}:\n", df[col].value_counts())
print("\nTarget 'Churn' distribution:")
if 'Churn' in df.columns:
    print(df['Churn'].value_counts())
