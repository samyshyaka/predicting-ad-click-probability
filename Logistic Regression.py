# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:00:49 2024

@author: Samy Shyaka
"""

# %%
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
import warnings
import pickle
import gc

# Suppress warnings
warnings.filterwarnings('ignore')

# %%

def reduce_mem_usage(df):
    """Reduce memory usage of a dataframe by downcasting numeric columns."""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.2f}% reduction)')
    return df

# Load and preprocess training data
print("Loading training data...")
train_data = pd.read_csv("ProjectTrainingData.csv")
if "id" in train_data.columns:
    train_data = train_data.drop(columns=["id"])

print("Reducing memory usage for training data...")
train_data = reduce_mem_usage(train_data)

# Process categorical columns
categorical_cols = [
    "site_id", "site_domain", "site_category", 
    "app_id", "app_domain", "app_category", 
    "device_id", "device_ip", "device_model",
    "banner_pos", "device_type", "device_conn_type"
]
categorical_cols = [col for col in categorical_cols if col in train_data.columns]

# Apply label encoding
label_encoders = {}
for col in categorical_cols:
    print(f"Encoding {col}...")
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

# Save preprocessed data
train_data.to_pickle('preprocessed_train.pkl')
np.save('label_encoders.npy', label_encoders, allow_pickle=True)

print("Preprocessing completed and saved!")

# %%

# Separate features and target
print("Preparing features and target...")
y = train_data["click"].values
X = train_data.drop(columns=["click"])
del train_data
gc.collect()

# Scale numerical features
print("Scaling features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Split data into train, validation, and test sets
print("Splitting dataset...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

del X, y, X_temp, y_temp
gc.collect()

# %%

# Train the logistic regression model on the training set
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict probabilities on the validation set
y_val_prob = log_reg.predict_proba(X_val)

# Calculate log loss on the validation set
val_log_loss = log_loss(y_val, y_val_prob)
print(f"Validation Log Loss: {val_log_loss:.4f}")

# Evaluate on the test set (after validation)
y_test_prob = log_reg.predict_proba(X_test)
test_log_loss = log_loss(y_test, y_test_prob)
print(f"Test Log Loss: {test_log_loss:.4f}")

# %%

# Load and preprocess test data
print("Loading test data...")
test_data = pd.read_csv("ProjectTestData.csv")

# Extract IDs for submission
if "id" in test_data.columns:
    test_ids = test_data["id"].values
    test_data = test_data.drop(columns=["id"])
else:
    raise ValueError("The 'id' column is missing from the test dataset.")

print("Reducing memory usage for test data...")
test_data = reduce_mem_usage(test_data)

# Apply label encoding and ignore rows with unseen labels
print("Applying label encoding to test data...")
valid_rows = np.ones(len(test_data), dtype=bool)  # Track valid rows

for col, le in label_encoders.items():
    if col in test_data.columns:
        known_classes = set(le.classes_)
        # Mark rows with unseen labels as invalid
        valid_rows &= test_data[col].isin(known_classes)

        # Apply label encoding only for valid rows
        test_data.loc[valid_rows, col] = le.transform(test_data.loc[valid_rows, col])

# Filter test_data and test_ids to keep only valid rows
test_data = test_data[valid_rows]
test_ids = test_ids[valid_rows]

# Convert to numpy and scale
print("Scaling test features...")
test_features = test_data.values
test_features = scaler.transform(test_features)
del test_data
gc.collect()

# Generate predictions on test data
print("Generating Test Predictions...")
y_test_pred = log_reg.predict_proba(test_features)[:, 1]  # Use test_features here

# Create submission file
print("Creating submission file...")
if len(test_ids) != len(y_test_pred):
    raise ValueError("Mismatch between test_ids and predictions length.")

submission = pd.DataFrame({
    "id": test_ids,
    "P(click)": y_test_pred
})

# Save the submission file
submission.to_csv('ProjectSubmission-Team13.csv', index=False)
print("Submission saved as 'ProjectSubmission-Team13.csv'")