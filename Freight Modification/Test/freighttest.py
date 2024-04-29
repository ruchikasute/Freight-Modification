# !pip install lazypredict xgboost lazypredict scipy

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from hdbcli import dbapi

# Define file paths
DATA_PATH = 'app/data/fredata.csv'
MODEL_PATH = 'app/model/xgb_model.pkl'

hana_connection = dbapi.connect(
    address="e7dcf416-50a3-4748-a430-913210d916ca.hna0.prod-us10.hanacloud.ondemand.com",
    port=443,
    user="CRAVE_INTERN#AZUREDATABASEUSER",
    password="eSn_L`xxB<A3T)[wi./bS:m+x(FJl^[6"
)

cursor = hana_connection.cursor()

# cursor.execute("DROP TABLE FREIGHT_01") 

def sendToHanaDB(schema:str, preprocessed_data:pd.DataFrame, tablename:str):
  cursor.execute(f"Set schema {schema}")
  try:
    try:
        create_query = "CREATE TABLE " + tablename + """(
    ID INT,
    Country FLOAT,
    "Fulfill Via" FLOAT,
    "Vendor INCO Term" FLOAT,
    "Shipment Mode" FLOAT,
    "Product Group" FLOAT,
    "Sub Classification" FLOAT,
    "Molecule/Test Type" FLOAT,
    Brand FLOAT,
    Dosage FLOAT,
    "Dosage Form" FLOAT,
    "Unit of Measure (Per Pack)" INT,
    "Line Item Quantity" INT,
    "Pack Price" FLOAT,
    "Unit Price" FLOAT,
    "First Line Designation" FLOAT,
    "Weight (Kilograms)" FLOAT,
    "Freight Cost (USD)" FLOAT,
    Shipping_cost FLOAT,
    "Predicted Shipping Cost" FLOAT)"""

        cursor.execute(create_query)        
        print(f"Table {tablename} created")
    except Exception as e:
        print(e)
    finally:
        final_data = [tuple(row) for row in preprocessed_data[['id','country','fulfill via','vendor inco term','shipment mode','product group','sub classification','molecule/test type','brand','dosage','dosage form','unit of measure (per pack)','line item quantity','pack price','unit price','first line designation','weight (kilograms)','freight cost (usd)','shipping_cost','predicted shipping cost']].itertuples(index=False)]
        sql_insert = f"INSERT INTO {tablename} (ID, Country, \"Fulfill Via\", \"Vendor INCO Term\", \"Shipment Mode\", \"Product Group\", \"Sub Classification\", \"Molecule/Test Type\", Brand, Dosage, \"Dosage Form\", \"Unit of Measure (Per Pack)\", \"Line Item Quantity\", \"Pack Price\", \"Unit Price\", \"First Line Designation\", \"Weight (Kilograms)\", \"Freight Cost (USD)\", Shipping_cost, \"Predicted Shipping Cost\") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        cursor.executemany(sql_insert, final_data)
        hana_connection.commit()

    return True
  except:
    return False

# Function to load data
def load_data(path):
    try:
        # Reading the file using pandas
        df = pd.read_csv(path)
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

# Preprocess data
data = load_data(DATA_PATH)
if data is not None:
    # Preprocess data
    preprocessed_data = preprocess_data(data)

    # Load the trained model using pickle
    with open(MODEL_PATH, 'rb') as f:
        xgb_model = pickle.load(f)

    # Extract features and target
    X_test = preprocessed_data.iloc[:, :-1]
    y_test = preprocessed_data['Shipping_cost']

    # Feature selection using mutual information
    scores = mutual_info_regression(X_test, y_test)
    k = 10
    top_k_features = X_test.columns[np.argsort(scores)[-k:]]
    X_test_top_k = X_test[top_k_features]

    # Make predictions
    predictions = xgb_model.predict(X_test_top_k)
    preprocessed_data['Predicted Shipping Cost'] = predictions

    preprocessed_data.columns = [col.lower() for col in preprocessed_data.columns]
    print(preprocessed_data.columns)

    # print( preprocessed_data)
    try:
        datastatus = sendToHanaDB("CRAVE_INTERN#AZUREDATABASEUSER", preprocessed_data, "FREIGHT_01")
        print(f"Status: {datastatus}")
    except Exception as e:
        print(e)

    cursor.close()
    hana_connection.close()
    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
else:
    print("Data preprocessing failed. Please check the data.")


