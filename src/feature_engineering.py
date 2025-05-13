import numpy as np

# Create new features based on existing data
def feature_engineering(data):
    # Time-Based Features
    data['Hour'] = (data['Time'] // 3600) % 24
    data['Is_Night'] = data['Hour'].apply(lambda x: 1 if 0 <= x <= 6 else 0)  # Night transactions
    
    # Amount-Based Features
    data['Log_Amount'] = np.log1p(data['Amount'])
    mean_amount = data['Amount'].mean()
    std_amount = data['Amount'].std()
    data['Amount_ZScore'] = (data['Amount'] - mean_amount) / std_amount
    
    # Statistical Features
    v_features = [f'V{i}' for i in range(1, 29)]
    data['V_Mean'] = data[v_features].mean(axis=1)
    data['V_Std'] = data[v_features].std(axis=1)
    
    # Drop original columns (Time and Amount)
    data = data.drop(['Time', 'Amount'], axis=1)
    
    return data
