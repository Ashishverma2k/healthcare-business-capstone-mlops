import sqlite3
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def build_features():
    print("Connecting to database...")
    conn = sqlite3.connect('healthcare_capstone.db')

    query = """
    SELECT 
        v.visit_id, v.patient_id, v.visit_date, v.department, v.visit_type, v.length_of_stay_hours, v.risk_score, v.doctor_id,
        p.age, p.gender, p.city, p.insurance_provider, p.chronic_flag, p.registration_date,
        b.bill_id, b.billed_amount, b.approved_amount, b.claim_status, b.payment_days, b.billing_date
    FROM visits v
    JOIN patients p ON v.patient_id = p.patient_id
    JOIN billing b ON v.visit_id = b.visit_id
    """
    df = pd.read_sql(query, conn)

    print("Cleaning data...")
    df['city'].fillna('Unknown', inplace=True)
    df['gender'].fillna('Unknown', inplace=True)
    
    median_age = df.loc[(df['age'] >= 0) & (df['age'] <= 120), 'age'].median()
    df['age'] = df['age'].apply(lambda x: median_age if (pd.isnull(x) or x < 0 or x > 120) else x)

    df = df[df['length_of_stay_hours'] > 0]
    df = df[df['billed_amount'] >= df['approved_amount']]

    print("Engineering features...")
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    df['patient_visit_count'] = df.groupby('patient_id')['visit_id'].transform('count')
    df['is_weekend_visit'] = df['visit_date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    
    rejection_rates = df[df['claim_status'] == 'Rejected'].groupby('insurance_provider').size() / df.groupby('insurance_provider').size()
    df['provider_rejection_rate'] = df['insurance_provider'].map(rejection_rates).fillna(0)

    print("Exporting model_table.csv...")
    df.to_csv('model_table.csv', index=False)
    print("Feature engineering pipeline complete.")

if __name__ == "__main__":
    build_features()
