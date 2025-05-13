import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess the data (features and target separation, scaling)
def preprocess_data(data):
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Scaling the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Handle imbalanced data with SMOTE
def handle_imbalance(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote

# Split data into train and test sets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
