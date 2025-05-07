from google.cloud import bigquery
import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
import joblib
from tqdm import tqdm  # For progress bar
from sklearn.model_selection import KFold

# Set up BigQuery client
client = bigquery.Client()

# Define your query
query = """
SELECT `Customer ID`, StockCode, Quantity, Description
FROM `retail-etl-456514.retail.sales`

"""

# Execute the query and load data into a pandas DataFrame
df = client.query(query).to_dataframe()
df.rename(columns={'Customer ID': 'customer_id'}, inplace=True)
df['customer_id'] = df['customer_id'].astype(int)

# Preview the data
print(df.head())

#----------------------user interaction
interaction_data = df[['customer_id', 'StockCode', 'Quantity']].copy()
df = df.dropna(subset=['Quantity'])  # Drop rows with NaN quantity

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df['user_idx'] = user_encoder.fit_transform(df['customer_id'])
df['item_idx'] = item_encoder.fit_transform(df['StockCode'])

# Convert indices to integers (required for the sparse matrix)
df['user_idx'] = df['user_idx'].astype(int)
df['item_idx'] = df['item_idx'].astype(int)

# Ensure that Quantity is a float (this can be integer, but casting it to float is a safe practice)
df['Quantity'] = df['Quantity'].astype(float)


interaction_matrix = coo_matrix(
    (df['Quantity'], (df['user_idx'], df['item_idx']))
)


print(interaction_matrix.shape)  # Verify the interaction matrix dimensions

#-----------------------item features
item_df = df[['StockCode', 'Description']].drop_duplicates()
item_df = item_df.dropna(subset=['Description'])  # drop null descriptions

item_df['item_idx'] = item_encoder.transform(item_df['StockCode'])
item_df = item_df.set_index('item_idx').sort_index()


# Convert descriptions to a sparse matrix using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
item_features = vectorizer.fit_transform(item_df['Description'])


print(item_features.shape)  # Verify the item feature matrix dimensions



#-------------------user features

user_df = df[['customer_id']]
#user_df['user_idx'] = user_encoder.transform(user_df['customer_id'])
user_df.loc[:, 'user_idx'] = user_encoder.transform(user_df['customer_id'])

user_df = user_df.set_index('user_idx').sort_index()

encoder = OneHotEncoder(sparse_output=True)
#user_features = encoder.fit_transform(user_df[['Country']])


def get_user_df():
    return user_df

#------------train
interaction_df2 = df[['user_idx', 'item_idx', 'Quantity']].copy()

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)
precisions = []
interaction_df = interaction_df2.sample(n=1000)  # Use a smaller sample of the data
kf = KFold(n_splits=5)  # Reduce the number of splits for testing

print(f"Running {K}-Fold Cross Validation...")

for fold, (train_index, test_index) in enumerate(kf.split(interaction_df), 1):
    train_data = interaction_df.iloc[train_index]
    test_data = interaction_df.iloc[test_index]
    
    # Create sparse matrices
    train_matrix = coo_matrix(
        (train_data['Quantity'], (train_data['user_idx'], train_data['item_idx']))
    )
    test_matrix = coo_matrix(
        (test_data['Quantity'], (test_data['user_idx'], test_data['item_idx']))
    )
    print("train shape: ",train_matrix.shape) 
    print("test shape: ",test_matrix.shape)


    # Train model
    model = LightFM(loss='warp')
    model.fit(train_matrix, epochs=10, num_threads=2)
    # Evaluate model
    precision = precision_at_k(
        model, test_matrix, k=5
    ).mean()
    
    print(f"Fold {fold} Precision@5: {precision:.4f}")
    precisions.append(precision)

# Final evaluation
average_precision = np.mean(precisions)
print(f"Average Precision@k=5 across {K} folds: {average_precision:.4f}")

# Save model and encoders
joblib.dump(model, 'model.pkl')
joblib.dump((user_encoder, item_encoder), 'encoders.pkl')
joblib.dump((item_features), 'features.pkl')

