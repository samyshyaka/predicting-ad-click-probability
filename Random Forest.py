# Step 1: Data Preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import gc
from sklearn.model_selection import train_test_split

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
train_data = pd.read_csv("/kaggle/input/ml-data/ProjectTrainingData.csv")
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

# Train-validation-test split
print("Splitting data into train, validation, and test sets...")
train_data, test_data = train_test_split(train_data, test_size=0.15, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.15 / 0.85, random_state=42)

# Save preprocessed data
train_data.to_pickle('/kaggle/working/preprocessed_train.pkl')
val_data.to_pickle('/kaggle/working/preprocessed_val.pkl')
test_data.to_pickle('/kaggle/working/preprocessed_test.pkl')
np.save('/kaggle/working/label_encoders.npy', label_encoders, allow_pickle=True)

print("Preprocessing completed and saved!")

# Step 2: Model Training
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import gc
import pickle
from sklearn.metrics import log_loss

# Load preprocessed data
print("Loading preprocessed data...")
train_data = pd.read_pickle('/kaggle/working/preprocessed_train.pkl')
val_data = pd.read_pickle('/kaggle/working/preprocessed_val.pkl')

# Split features and target
print("Preparing features and target...")
train_target = train_data["click"].values
train_features = train_data.drop(columns=["click"]).values
val_target = val_data["click"].values
val_features = val_data.drop(columns=["click"]).values

del train_data, val_data
gc.collect()

# Scale features
print("Scaling features...")
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)

# Save scaler
with open('/kaggle/working/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Train model
print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,  # Reduced for testing
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

rf_model.fit(train_features, train_target)

# Validate model
print("Validating model...")
val_predictions = rf_model.predict_proba(val_features)[:, 1]
val_log_loss = log_loss(val_target, val_predictions)
print(f"Log Loss on Validation Data: {val_log_loss}")

# Save model
print("Saving model...")
with open('/kaggle/working/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Load preprocessed test data
print("Loading preprocessed test data...")
test_data = pd.read_pickle('/kaggle/working/preprocessed_test.pkl')

# Split features and target
print("Preparing test features and target...")
test_target = test_data["click"].values
test_features = test_data.drop(columns=["click"]).values

# Clean up memory
del test_data
gc.collect()

# Load the saved scaler
print("Loading the saved scaler...")
with open('/kaggle/working/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Scale test features
print("Scaling test features...")
test_features = scaler.transform(test_features)

# Test the model
print("Loading the trained model...")
with open('/kaggle/working/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

print("Generating test predictions...")
test_predictions = rf_model.predict_proba(test_features)[:, 1]

# Calculate log loss on test data
print("Calculating Log Loss on test data...")
test_log_loss = log_loss(test_target, test_predictions)
print(f"Log Loss on Test Data: {test_log_loss}")

# Step 3: Prediction
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import gc

# Load the model and scaler
print("Loading model and scaler...")
with open('/kaggle/working/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('/kaggle/working/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load label encoders
label_encoders = np.load('/kaggle/working/label_encoders.npy', allow_pickle=True).item()

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
    return df

# Load and preprocess test data
print("Loading test data...")
test_data = pd.read_csv("/kaggle/input/ml-data/ProjectTestData.csv")
if "id" in test_data.columns:
    ids = test_data["id"].values
    test_data = test_data.drop(columns=["id"])
else:
    ids = np.arange(len(test_data))

print("Reducing memory usage for test data...")
test_data = reduce_mem_usage(test_data)

# Apply label encoding and handle unseen labels
print("Applying label encoding to test data...")
for col, le in label_encoders.items():
    if col in test_data.columns:
        known_classes = set(le.classes_)
        test_data[col] = test_data[col].apply(lambda x: x if x in known_classes else '-1')

        # Add the unseen label placeholder if not already in classes
        if '-1' not in le.classes_:
            le.classes_ = np.append(le.classes_, '-1')

        # Transform the column
        test_data[col] = le.transform(test_data[col])

# Convert to numpy and scale
print("Scaling test features...")
test_features = test_data.values
test_features = scaler.transform(test_features)
del test_data
gc.collect()

# Generate predictions
print("Generating predictions...")
test_predictions = rf_model.predict_proba(test_features)[:, 1]

# Create submission file
print("Creating submission file...")
submission = pd.DataFrame({"id": ids, "P(click)": test_predictions})
submission.to_csv("/kaggle/working/ProjectSubmission-TeamX.csv", index=False)

print("Prediction completed and submission file created!")
