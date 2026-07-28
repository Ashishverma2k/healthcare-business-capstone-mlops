import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def validate_data(new_data, reference_data):
    # Data Validation: Check for missing values and unseen categories
    missing = new_data.isnull().sum()
    unseen = [col for col in new_data.columns if new_data[col].dtype == 'object' 
              and not set(new_data[col].unique()).issubset(reference_data[col].unique())]
    
    # Drift Detection: Kolmogorov-Smirnov test for feature drift
    drift = {}
    for col in reference_data.select_dtypes(include=[np.number]).columns:
        stat, p = ks_2samp(reference_data[col], new_data[col])
        drift[col] = "Drift Detected" if p < 0.05 else "Stable"
    
    return {"Missing": missing.to_dict(), "Unseen_Categories": unseen, "Drift": drift}