#!/usr/bin/env python
# coding: utf-8

# ## Import All Libraries

# In[1]:


# Standard Libraries
import pandas as pd
import numpy as np

# Machine Learning Models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
from sklearn.metrics import roc_auc_score, roc_curve

# Handling Imbalanced Data
from imblearn.over_sampling import SMOTE

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load and Explore Dataset

# In[3]:


# Load dataset (Kaggle se download karo: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
data = pd.read_csv('creditcard.csv')


# In[4]:


# Pehle 5 rows dikhao
display(data.head())


# In[5]:


# Dataset basic info
print("\nDataset Info:")
data.info()


# In[6]:


# Class distribution check 
print("\nClass Distribution (0 = Normal, 1 = Fraud):")
print(data['Class'].value_counts())


# In[7]:


# Missing values check
print("\nMissing Values:")
print(data.isnull().sum())


# ## Feature Engineering

# In[9]:


# 1. Time-Based Features
data['Hour'] = (data['Time'] // 3600) % 24  # Convert seconds to hour of day
data['Is_Night'] = data['Hour'].apply(lambda x: 1 if 0 <= x <= 6 else 0)  # Night transactions (12 AM - 6 AM)


# In[10]:


# 2. Amount-Based Features
data['Log_Amount'] = np.log1p(data['Amount'])  # Log transform to reduce skewness
mean_amount = data['Amount'].mean()
std_amount = data['Amount'].std()
data['Amount_ZScore'] = (data['Amount'] - mean_amount) / std_amount  # Z-score of amount


# In[11]:


# 3. Statistical Features
v_features = [f'V{i}' for i in range(1, 29)]
data['V_Mean'] = data[v_features].mean(axis=1)
data['V_Std'] = data[v_features].std(axis=1)


# In[12]:


data.columns


# In[13]:


# Drop original Time and Amount (replaced by engineered features)
data = data.drop(['Time', 'Amount'], axis=1)


# ## Data Preprocessing

# In[15]:


# Features aur target alag karo
X = data.drop('Class', axis=1)
y = data['Class']


# In[16]:


data.duplicated().any()


# In[17]:


print("\nNumber of duplicated:")
data.duplicated().sum()


# In[18]:


# Remove duplication
new_data = data.drop_duplicates()
new_data.shape


# In[19]:


new_data.columns


# In[20]:


new_data.duplicated().any()


# In[21]:


# Class distribution 
print("\nClass Distribution (0 = Normal, 1 = Fraud):")
print(new_data['Class'].value_counts())


# In[22]:


# Scaling karo (Amount aur Time ko scale karna zaroori hai)
from sklearn.preprocessing import StandardScaler
# Scaling karo (new numerical features ko scale karna zaroori hai)
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[23]:


# Data ko train-test mein split karo (80-20 ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Dono sets mein same ratio maintain karo
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


# ## Handle Imbalanced Data (SMOTE)

# In[25]:


print("Original Class Distribution in Training Set:")
print(y_train.value_counts())

# SMOTE apply karo taki fraud cases artificially increase ho
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("After SMOTE, Class Distribution:\n", pd.Series(y_train_smote).value_counts())


# ## Data Visualization

# In[27]:


import numpy as np

normal_records = new_data.Class == 0
fraud_records = new_data.Class == 1

num_cols = len(new_data.columns)
rows = int(np.ceil(num_cols / 3)) 
plt.figure(figsize=(15, rows * 3))

for n, col in enumerate(new_data.columns):
    plt.subplot(rows, 3, n + 1)  
    if new_data[col].dtype in ['int64', 'float64']: 
        sns.histplot(new_data.loc[normal_records, col], color='green', kde=True, stat="density", label='Normal', alpha=0.6)
        sns.histplot(new_data.loc[fraud_records, col], color='red', kde=True, stat="density", label='Fraud', alpha=0.6)
        plt.title(col, fontsize=15)
        plt.legend()
        
plt.tight_layout()
plt.show()


# In[28]:


custom_palette = sns.color_palette(["blue", "gold"])
plt.figure(figsize=(8, 6))
sns.countplot(x="Class", data=new_data, palette=custom_palette)
plt.title("Class Distribution")
plt.show()


# ## Distribution of Classes After Resampling (SMOTE)

# In[30]:


custom_palette = sns.color_palette(["blue", "gold"])
plt.figure(figsize=(8, 6))
sns.countplot(x=y_train_smote, palette=custom_palette)
plt.title("Distribution of Classes After Resampling (SMOTE)")
plt.xlabel("Class (0: Fraud, 1: Legitimate)")
plt.ylabel("Count")
plt.show()


# #                          **Model Building**

# ## Logistic Regression

# In[33]:


# Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_smote, y_train_smote)

# Predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluation
print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lr):.4f}")


# In[34]:


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_lr)
labels = ['Non-Fraudulent', 'Fraudulent']

plt.figure(figsize=(6, 4))
ax = sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels)

# Annotate with TP, TN, etc.
matrix_labels = [['TN', 'FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        value = cm[i][j]
        label = matrix_labels[i][j]
        ax.text(j + 0.5, i + 0.5, f'{label} = {value}', 
                ha='center', va='center', color='black', fontsize=12)

plt.title('Logistic Regression - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()


# In[35]:


# Logistic Regression ROC-AUC
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
roc_auc_lr = roc_auc_score(y_test, y_prob_lr)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_prob_lr)

print("Logistic Regression ROC-AUC Score:", roc_auc_lr)

plt.figure(figsize=(8,6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.show()


# ## Decision Tree

# In[37]:


# Decision Tree Model
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train_smote, y_train_smote)

# Predictions
y_pred_dt = dt_model.predict(X_test)

# Evaluation
print("Decision Tree Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_dt):.4f}")


# In[38]:


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_dt)
labels = ['Non-Fraudulent', 'Fraudulent']

plt.figure(figsize=(6, 4))
ax = sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels)

# Annotate with TP, TN, etc.
matrix_labels = [['TN', 'FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        value = cm[i][j]
        label = matrix_labels[i][j]
        ax.text(j + 0.5, i + 0.5, f'{label} = {value}', 
                ha='center', va='center', color='black', fontsize=12)

plt.title('Logistic Regression - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()


# In[39]:


# Decision Tree ROC-AUC
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]
roc_auc_dt = roc_auc_score(y_test, y_prob_dt)
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, y_prob_dt)

print("Decision Tree ROC-AUC Score:", roc_auc_dt)

plt.figure(figsize=(8,6))
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc='lower right')
plt.show()


# ## Random Forest

# In[41]:


# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation
print("Random Forest Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")


# In[42]:


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
labels = ['Non-Fraudulent', 'Fraudulent']

plt.figure(figsize=(6, 4))
ax = sns.heatmap(cm, annot=False, cmap='Oranges', xticklabels=labels, yticklabels=labels)

# Annotate with TP, TN, etc.
matrix_labels = [['TN', 'FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        value = cm[i][j]
        label = matrix_labels[i][j]
        ax.text(j + 0.5, i + 0.5, f'{label} = {value}', 
                ha='center', va='center', color='black', fontsize=12)

plt.title('Random Forest - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()


# In[75]:


# Random Forest ROC-AUC
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf)

print("Random Forest ROC-AUC Score:", roc_auc_rf)

plt.figure(figsize=(8,6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.show()


# ## XGBoost

# In[77]:


# XGBoost Model
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_smote, y_train_smote)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluation
print("XGBoost Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_xgb):.4f}")


# In[78]:


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_xgb)
labels = ['Non-Fraudulent', 'Fraudulent']

plt.figure(figsize=(6, 4))
ax = sns.heatmap(cm, annot=False, cmap='Purples', xticklabels=labels, yticklabels=labels)

# Annotate with TP, TN
matrix_labels = [['TN', 'FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        value = cm[i][j]
        label = matrix_labels[i][j]
        ax.text(j + 0.5, i + 0.5, f'{label} = {value}', 
                ha='center', va='center', color='black', fontsize=12)

plt.title('XGBoos - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()


# In[79]:


# XGBoost ROC-AUC
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
roc_auc_xgb = roc_auc_score(y_test, y_prob_xgb)
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_prob_xgb)

print("XGBoost ROC-AUC Score:", roc_auc_xgb)

plt.figure(figsize=(8,6))
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.legend(loc='lower right')
plt.show()


# ## Neural Network

# In[99]:


#Scale the data (Neural Networks need scaled features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)  # Using SMOTE-resampled training data
X_test_scaled = scaler.transform(X_test)  # Original test data (not resampled)
#Neural Network Model
nn_model = MLPClassifier(
    hidden_layer_sizes=(50,),  # Single hidden layer with 50 neurons
    activation='relu',         # Rectified Linear Unit activation
    solver='adam',            # Optimizer
    max_iter=1000,            # Maximum iterations
    random_state=42,
    verbose=True              # Shows training progress
)
# Train the model
print("Training Neural Network...")
nn_model.fit(X_train_scaled, y_train_smote)  # Using resampled labels
# Cell 3: Predictions and Evaluation
y_pred_nn = nn_model.predict(X_test_scaled)  # Using scaled test data

print("\nNeural Network Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nn):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_nn):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_nn):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_nn):.4f}")


# In[109]:


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_nn)
labels = ['Non-Fraudulent', 'Fraudulent']

plt.figure(figsize=(6, 4))
ax = sns.heatmap(cm, annot=False, cmap='Purples', xticklabels=labels, yticklabels=labels)

# Annotate with TP, TN
matrix_labels = [['TN', 'FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        value = cm[i][j]
        label = matrix_labels[i][j]
        ax.text(j + 0.5, i + 0.5, f'{label} = {value}', 
                ha='center', va='center', color='black', fontsize=12)

plt.title('Neural Network - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()


# In[111]:


# ROC Curve and AUC Score (Additional Evaluation)
from sklearn.metrics import roc_auc_score, roc_curve

y_prob_nn = nn_model.predict_proba(X_test_scaled)[:, 1]
nn_auc = roc_auc_score(y_test, y_prob_nn)
fpr, tpr, _ = roc_curve(y_test, y_prob_nn)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Neural Network (AUC = {nn_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Neural Network')
plt.legend(loc='lower right')
plt.show()


# # Model Comparison

# In[117]:


# Sabhi models ke results collect karo
results = [
    {'Model': 'Logistic Regression', 'Accuracy': accuracy_score(y_test, y_pred_lr),
     'Precision': precision_score(y_test, y_pred_lr), 'Recall': recall_score(y_test, y_pred_lr),
     'F1-Score': f1_score(y_test, y_pred_lr)},
    
    {'Model': 'Decision Tree', 'Accuracy': accuracy_score(y_test, y_pred_dt),
     'Precision': precision_score(y_test, y_pred_dt), 'Recall': recall_score(y_test, y_pred_dt),
     'F1-Score': f1_score(y_test, y_pred_dt)},
    
    {'Model': 'Random Forest', 'Accuracy': accuracy_score(y_test, y_pred_rf),
     'Precision': precision_score(y_test, y_pred_rf), 'Recall': recall_score(y_test, y_pred_rf),
     'F1-Score': f1_score(y_test, y_pred_rf)},
    
    {'Model': 'XGBoost', 'Accuracy': accuracy_score(y_test, y_pred_xgb),
     'Precision': precision_score(y_test, y_pred_xgb), 'Recall': recall_score(y_test, y_pred_xgb),
     'F1-Score': f1_score(y_test, y_pred_xgb)},
    
    {'Model': 'Neural Network', 'Accuracy': accuracy_score(y_test, y_pred_nn),
     'Precision': precision_score(y_test, y_pred_nn), 'Recall': recall_score(y_test, y_pred_nn),
     'F1-Score': f1_score(y_test, y_pred_nn)}
]

# DataFrame mein convert karo
results_df = pd.DataFrame(results)

# Results display karo
print("Model Comparison:")
display(results_df)

# Visualization
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=45)
plt.ylim(0.9, 1.0)

plt.subplot(1,2,2)
sns.barplot(x='Model', y='F1-Score', data=results_df)
plt.title('Model F1-Score Comparison')
plt.xticks(rotation=45)
plt.ylim(0.0, 1.0)

plt.tight_layout()
plt.show()


# In[119]:


# All models comparison ROC curve
plt.figure(figsize=(10,8))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
plt.plot(fpr_nn, tpr_nn, label=f'Neural Network (AUC = {roc_auc_nn:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




