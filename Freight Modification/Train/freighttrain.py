# !pip install lazypredict xgboost lazypredict scipy

# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.impute import KNNImputer
import pickle

# Define file paths
DATA_PATH = 'app/data/fredata.csv'
MODEL_PATH = 'app/model/xgb_model.pkl'

# Function to load data
def load_data(DATA_PATH):
    try:
        # Reading the file using pandas
        df = pd.read_csv(DATA_PATH)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        # If there's an error reading the file, print "Error" along with the error message
        print("Error:", e)
        return None

# Function to preprocess data
def preprocess_data(df):
    # Handle missing values in 'Line Item Insurance (USD)' column
    df['Line Item Insurance (USD)'].fillna(0.1, inplace=True)

    # For 'Weight_new' column
    df['Weight_new'] = np.where(df['Weight (Kilograms)'] == "Weight Captured Separately", df['Weight (Kilograms)'],
                                 np.where(df['Weight (Kilograms)'].str.split(" ").str[0] == "See", "See ASN/DN Tag", "Normal Measurement"))

    # For 'Freight_cost_new' column
    df['Freight_cost_new'] = np.where(df['Freight Cost (USD)'].isin(["Freight Included in Commodity Cost", 'Invoiced Separately']), df['Freight Cost (USD)'],
                                       np.where(df['Freight Cost (USD)'].str.split(" ").str[0] == "See", "See ASN/DN Tag", "Normal Measurement"))

    # Convert to datetime format
    for col in ['Scheduled Delivery Date', 'Delivered to Client Date']:
        df[col] = pd.to_datetime(df[col], infer_datetime_format=True)

    # Convert to Numeric
    for col in ['Weight (Kilograms)', 'Freight Cost (USD)']:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop specified columns
    columns_to_drop = ['Project Code', 'PQ #', 'PO / SO #', 'ASN/DN #', "PQ First Sent to Client Date", "PO Sent to Vendor Date",
                       'Delivery Recorded Date', 'Managed By', 'Item Description']
    if all(col in df.columns for col in columns_to_drop):
        df.drop(columns=columns_to_drop, inplace=True)
        print("Columns dropped successfully.")
    else:
        print("One or more columns to be dropped do not exist in the DataFrame.")

    # Replace 'N/A - From RDC' with 'RDC'
    df.replace('N/A - From RDC', 'RDC', inplace=True)

    # Random Sample Imputation for Categorical Columns
    for col in ['Dosage', 'Shipment Mode']:
        rand_samples = df[col].dropna().sample(df[col].isnull().sum())
        rand_samples.index = df[df[col].isnull()].index
        df.loc[df[col].isnull(), col] = rand_samples

    # KNN Imputation for numerical columns
    for col in ['Weight (Kilograms)', 'Freight Cost (USD)']:
        imputer = KNNImputer(n_neighbors=10)
        df[col] = imputer.fit_transform(df[[col]])

    # Calculate 'Shipping_cost' column
    df['Shipping_cost'] = df['Freight Cost (USD)'] + df['Line Item Value'] + df['Line Item Insurance (USD)']

    # Label encoding and normalization for categorical columns
    cl_categ = df.dtypes[df.dtypes == 'object'].index
    cl_categ = cl_categ.drop(['Vendor', 'Weight_new', 'Freight_cost_new'])
    lab_en = LabelEncoder()
    scaler = MinMaxScaler()
    for col in cl_categ:
        df[col] = lab_en.fit_transform(df[col]) #Label Encoding
        df[col] = scaler.fit_transform(np.array(df[col]).reshape(-1, 1)) #Normalizing the data

    # Drop specified columns
    df.drop(['Line Item Insurance (USD)', 'Line Item Value', 'Scheduled Delivery Date', 'Vendor', 'Delivered to Client Date',
             'Manufacturing Site', 'Weight_new', 'Freight_cost_new'], axis=1, inplace=True)

    return df

# Load data
data = load_data(DATA_PATH)

if data is not None:
    # Preprocess data
    preprocessed_data = preprocess_data(data)

    # Split data into features and target
    X = preprocessed_data.iloc[:, :-1]
    y = preprocessed_data['Shipping_cost']

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.feature_selection import mutual_info_regression
# Feature selection using mutual information
scores = mutual_info_regression(X_train, y_train)
k = 10
top_k_features = X.columns[np.argsort(scores)[-k:]]
X_top_k = X[top_k_features]

# Train XGBoost Regressor
xgb_model = XGBRegressor()
xgb_model.fit(X_top_k, y)

# Save the trained model using pickle
with open(MODEL_PATH, 'wb') as f:
        pickle.dump(xgb_model, f)
        print("Model saved successfully.")


# DATA_PATH = 'app/data/fredata.csv'
# df = pd.read_csv(DATA_PATH)
# print(df.shape)