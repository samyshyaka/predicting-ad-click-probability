from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss_and_gradient(theta, X, y):
    m = X.shape[0]
    predictions = sigmoid(X @ theta)
    loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    gradient = (1 / m) * (X.T @ (predictions - y))
    return loss, gradient

#Calculate weights
def optimize_theta(X, y):
    initial_theta = np.zeros(X.shape[1])
    options = {'maxiter': 1000}
    
    result = minimize(
        fun=lambda t: compute_loss_and_gradient(t, X, y)[0],
        x0=initial_theta,
        jac=lambda t: compute_loss_and_gradient(t, X, y)[1],
        method='L-BFGS-B',
        options=options
    )
    return result.x

# Function to reduce memory usage
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
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Mem. usage decreased to {end_mem:.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv('ProjectTrainingData.csv')
    df = reduce_mem_usage(df)

    # Encode categorical columns
    categorical_columns = ['site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category',
                           'device_id', 'device_ip', 'device_model', 'C1', 'banner_pos', 'device_type',
                           'device_conn_type'] + [f'C{i}' for i in range(14, 22)]
    encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col].astype(str))

    # Define feature matrix (X) and target variable (y)
    X = df.drop(['click'], axis=1).values
    y = df['click'].values

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Add intercept
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_val = np.c_[np.ones(X_val.shape[0]), X_val]
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]

    # Train logistic regression using gradient descent
    theta = optimize_theta(X_train, y_train)

    # Make predictions
    y_val_pred = sigmoid(X_val @ theta)
    y_test_pred = sigmoid(X_test @ theta)

    # Calculate and print log loss
    print('Validation Log Loss:', log_loss(y_val, y_val_pred))
    print('Test Log Loss:', log_loss(y_test, y_test_pred))