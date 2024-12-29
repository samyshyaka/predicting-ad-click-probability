# Step 1: Data Preprocessing
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers.legacy import Adam  # Use legacy optimizer for better performance on M1/M2 Macs
from tensorflow.keras.models import load_model

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
train_data = pd.read_csv("Project Data/ProjectTrainingData.csv")
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
train_data.to_pickle('Project Data/preprocessed_train.pkl')
np.save('Project Data/label_encoders.npy', label_encoders, allow_pickle=True)

print("Preprocessing completed and saved!")


# Step 2: Model Training
# Load and preprocess training data
print("Loading training data...")
train_data = pd.read_csv('Project Data/ProjectTrainingData.csv')
if "id" in train_data.columns:
    train_data = train_data.drop(columns=["id"])

print("Reducing memory usage for training data...")
train_data = reduce_mem_usage(train_data)

# Separate features and target
print("Preparing features and target...")
y = train_data["click"].values
X = train_data.drop(columns=["click"])
del train_data
gc.collect()

# Process categorical columns with label encoding
print("Encoding categorical features...")
categorical_cols = [
    "site_id", "site_domain", "site_category", 
    "app_id", "app_domain", "app_category", 
    "device_id", "device_ip", "device_model",
    "banner_pos", "device_type", "device_conn_type"
]
categorical_cols = [col for col in categorical_cols if col in X.columns]
label_encoders = {}

for col in categorical_cols:
    print(f"Encoding {col}...")
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Save label encoders for future use
with open('Project Data/label_encoders.npy', 'wb') as f:
    np.save(f, label_encoders, allow_pickle=True)

# Scale numerical features
print("Scaling features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler
with open('Project Data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Split data into train, validation, and test sets
print("Splitting dataset...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

del X, y, X_temp, y_temp
gc.collect()

# Build the neural network
print("Building the neural network...")
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
print("Compiling the model with legacy optimizer...")
model.compile(optimizer=Adam(learning_rate=0.001),  # Use legacy Adam optimizer
              loss='binary_crossentropy',
              metrics=['accuracy', AUC()])


# Train the model
print("Training the neural network...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=64,
    verbose=1
)

# Save the model
print("Saving the model...")
model.save('Project Data/nn_model.h5')

# Save training history
with open('Project Data/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

print("Model training completed!")

print("Evaluating the model on the test set and validation set (log-loss)...")
test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test)
print(f"Test Log-Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test AUC: {test_auc:.4f}")
val_loss, val_accuracy, val_auc = model.evaluate(X_val, y_val)
print(f"Validation Log-Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation AUC: {val_auc:.4f}")

# Step 3: Prediction

# Load the model and scaler
print("Loading neural network model and scaler...")
nn_model = load_model('Project Data/nn_model.h5')
with open('Project Data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load label encoders
label_encoders = np.load('Project Data/label_encoders.npy', allow_pickle=True).item()

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
test_data = pd.read_csv("Project Data/ProjectTestData.csv")

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

# Generate predictions
print("Generating predictions...")
test_predictions = nn_model.predict(test_features).flatten()  # Flatten to get a 1D array

# Create submission file
print("Creating submission file...")
submission = pd.DataFrame({
    "id": test_ids,
    "P(click)": test_predictions
})
submission.to_csv("Project Data/ProjectSubmission-TeamX.csv", index=False)

print("Prediction completed and submission file created!")
