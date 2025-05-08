
import pandas as pd
from google.cloud import bigquery
from scipy.sparse import vstack, coo_matrix
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k
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
df['customer_id'] = df['customer_id'].astype(str)  # Convert to string to handle potential non-numeric IDs

# Preview the data
print(df.head())

# Clean data - drop missing values
df = df.dropna(subset=['Quantity', 'customer_id', 'StockCode'])

# Create unique indices for users and items
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

# Fit encoders on the entire dataset to ensure consistent mappings
unique_users = df['customer_id'].unique()
unique_items = df['StockCode'].unique()

user_encoder.fit(unique_users)
item_encoder.fit(unique_items)

# Transform the data
df['user_idx'] = user_encoder.transform(df['customer_id'])
df['item_idx'] = item_encoder.transform(df['StockCode'])

# Convert indices to integers (required for the sparse matrix)
df['user_idx'] = df['user_idx'].astype(int)
df['item_idx'] = df['item_idx'].astype(int)

# Ensure that Quantity is a float
df['Quantity'] = df['Quantity'].astype(float)

# Get shape dimensions for matrices
n_users = len(user_encoder.classes_)
n_items = len(item_encoder.classes_)
print(f"Number of users: {n_users}, Number of items: {n_items}")

# Create full interaction matrix (will be used for final model training)
interaction_matrix = coo_matrix(
    (df['Quantity'], (df['user_idx'], df['item_idx'])),
    shape=(n_users, n_items)
)

print(f"Interaction matrix shape: {interaction_matrix.shape}")

# Process item features
print("Processing item features...")
item_df = df[['StockCode', 'Description']].drop_duplicates()
item_df = item_df.dropna(subset=['Description'])  # drop null descriptions

# Use the same encoder that was fit on the entire dataset
item_df['item_idx'] = item_encoder.transform(item_df['StockCode'])
item_df = item_df.set_index('item_idx').sort_index()

# Convert descriptions to a sparse matrix using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
item_features = vectorizer.fit_transform(item_df['Description'])

print(f"Item features shape: {item_features.shape}")

# Make sure item features has the right number of rows
if item_features.shape[0] < n_items:
    # For items without descriptions, we'll use zero vectors
    missing_items = n_items - item_features.shape[0]
    print(f"Adding {missing_items} zero vectors for items without descriptions")
    zeros = coo_matrix((missing_items, item_features.shape[1]))
    item_features = vstack([item_features, zeros])

# User features preparation
print("Processing user features...")
user_df = df[['customer_id', 'user_idx']].drop_duplicates()
user_df = user_df.set_index('user_idx').sort_index()

# Since we don't have user features in the query, we'll create a simple identity matrix
# This effectively gives each user a unique one-hot encoded feature
user_features = coo_matrix(np.eye(n_users))
print(f"User features shape: {user_features.shape}")

# Cross-validation setup
print("Setting up cross-validation...")
interaction_df = df[['user_idx', 'item_idx', 'Quantity']].copy()

# For testing purposes, we can use a smaller sample
# Comment this line for full dataset training
#sample_size = min(100000, len(interaction_df))  # Use at most 10,000 interactions
#interaction_df = interaction_df.sample(n=sample_size, random_state=42)

K = 3
kf = KFold(n_splits=K, shuffle=True, random_state=42)
precisions = []
recalls=[]
f1s=[]
print(f"Running {K}-Fold Cross Validation...")

# Hyperparameters for the model
num_components = 128  # Higher dimensionality for latent factors
learning_rate = 0.05
epochs = 30
regularization = 0.001


for fold, (train_index, test_index) in enumerate(kf.split(interaction_df), 1):
    print(f"\nFold {fold}/{K}")
    train_data = interaction_df.iloc[train_index]
    test_data = interaction_df.iloc[test_index]
    
    # Create sparse matrices with correct dimensions
    train_matrix = coo_matrix(
        (train_data['Quantity'], (train_data['user_idx'], train_data['item_idx'])),
        shape=(n_users, n_items)
    )
    test_matrix = coo_matrix(
        (test_data['Quantity'], (test_data['user_idx'], test_data['item_idx'])),
        shape=(n_users, n_items)
    )
    
    print(f"Train matrix shape: {train_matrix.shape}")
    print(f"Test matrix shape: {test_matrix.shape}")

    # Train model
    model = LightFM(
        no_components=num_components,
        learning_rate=learning_rate,
        loss='warp',  # WARP loss for better ranking
        item_alpha=regularization,  # L2 penalty on item features
        user_alpha=regularization   # L2 penalty on user features
    )
    
    # Train with item and user features
    model.fit(
        train_matrix, 
        user_features=user_features,
        item_features=item_features,
        epochs=30, 
        num_threads=2,
        verbose=True
    )
    
    # Evaluate model
    precision = precision_at_k(
        model, 
        test_matrix, 
        user_features=user_features,
        item_features=item_features,
        k=5
    ).mean()

    recall= recall_at_k(model, test_matrix, user_features=user_features,
        item_features=item_features,k=5).mean()
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Fold {fold} Precision@5: {precision:.4f}")
    print(f"Fold {fold} recall@5: {recall:.4f}")
    print(f"Fold {fold} f1@5: {f1:.4f}")

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
# Final evaluation
average_precision = np.mean(precisions)
average_recall = np.mean(recalls)
average_f1 = np.mean(f1s)

print(f"Average Precision@k=5 across {K} folds: {average_precision:.4f}")
print(f"Average recalln@k=5 across {K} folds: {average_recall:.4f}")
print(f"Average f1@k=5 across {K} folds: {average_f1:.4f}")


# Save model and encoders
print("Saving model and related data...")
joblib.dump(model, 'model2.pkl')
joblib.dump((user_encoder, item_encoder), 'encoders2.pkl')
joblib.dump(item_features, 'item_features2.pkl')
joblib.dump(user_features, 'user_features2.pkl')
joblib.dump(vectorizer, 'vectorizer2.pkl')
