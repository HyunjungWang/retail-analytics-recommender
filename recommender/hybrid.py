from google.cloud import bigquery
import pandas as pd
from scipy.sparse import vstack, coo_matrix
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k
import joblib
from tqdm import tqdm

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
df['customer_id'] = df['customer_id'].astype(str)

# Clean data
df = df.dropna(subset=['Quantity', 'customer_id', 'StockCode'])

# Filter out cancelled orders (negative quantities) and returns
df = df[df['Quantity'] > 0]

# Filter out users with very few interactions (cold-start problem)
user_counts = df['customer_id'].value_counts()
min_user_interactions = 5  # Minimum number of interactions per user
valid_users = user_counts[user_counts >= min_user_interactions].index
df = df[df['customer_id'].isin(valid_users)]

# Create unique indices for users and items
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

# Fit encoders on the entire dataset
unique_users = df['customer_id'].unique()
unique_items = df['StockCode'].unique()

user_encoder.fit(unique_users)
item_encoder.fit(unique_items)

# Transform the data
df['user_idx'] = user_encoder.transform(df['customer_id'])
df['item_idx'] = item_encoder.transform(df['StockCode'])

# Convert indices to integers
df['user_idx'] = df['user_idx'].astype(int)
df['item_idx'] = df['item_idx'].astype(int)
df['Quantity'] = df['Quantity'].astype(float)


# Get shape dimensions for matrices
n_users = len(user_encoder.classes_)
n_items = len(item_encoder.classes_)
print(f"Number of users: {n_users}, Number of items: {n_items}")

# Create full interaction matrix
interaction_matrix = coo_matrix(
    (df['Quantity'], (df['user_idx'], df['item_idx'])),
    shape=(n_users, n_items)
)

print(f"Interaction matrix shape: {interaction_matrix.shape}")

# Process item features
print("Processing item features...")
item_df = df[['StockCode', 'Description']].drop_duplicates()
item_df = item_df.dropna(subset=['Description'])

# Add the item index
item_df['item_idx'] = item_encoder.transform(item_df['StockCode'])
item_df = item_df.set_index('item_idx').sort_index()

# Enhanced text features using TF-IDF with better parameters
vectorizer = TfidfVectorizer(
    stop_words='english',
    min_df=5,  # Ignore terms that appear in less than 5 documents
    max_df=0.7,  # Ignore terms that appear in more than 70% of documents
    ngram_range=(1, 2)  # Use both unigrams and bigrams
)
item_features = vectorizer.fit_transform(item_df['Description'])

print(f"Item features shape: {item_features.shape}")

# Make sure item features has the right number of rows
if item_features.shape[0] < n_items:
    missing_items = n_items - item_features.shape[0]
    print(f"Adding {missing_items} zero vectors for items without descriptions")
    zeros = coo_matrix((missing_items, item_features.shape[1]))
    item_features = vstack([item_features, zeros])

# User features preparation
print("Processing user features...")

# Create user purchase history features 
user_purchase_history = interaction_matrix.copy()
# Convert to binary matrix (purchased or not)
user_purchase_history.data = np.ones_like(user_purchase_history.data)

# Create user features from their purchase history
user_features = user_purchase_history
print(f"User features shape: {user_features.shape}")

# Implement leave-one-out cross validation
print("Setting up train-test split...")

# Set the train-test ratio (90-10 split)
train_ratio = 0.9  # Can be changed to 0.8 for 80-20 split

# Group by user to ensure we have both train and test data for each user
user_item_df = df[['user_idx', 'item_idx', 'Quantity']].copy()

# Create train and test sets by splitting interactions for each user
train_data = []
test_data = []

print("Creating train-test split by user...")
for user_id, user_data in tqdm(user_item_df.groupby('user_idx')):
    # Shuffle user data
    user_interactions = user_data.sample(frac=1, random_state=42)
    
    # Split interactions
    n_interactions = len(user_interactions)
    n_train = int(n_interactions * train_ratio)
    
    # Ensure at least one interaction in test set if user has enough data
    if n_interactions > 1:
        train_data.append(user_interactions.iloc[:n_train])
        test_data.append(user_interactions.iloc[n_train:])
    else:
        # If user has only one interaction, put it in train set
        train_data.append(user_interactions)

# Combine all users' train and test data
train_df = pd.concat(train_data)
test_df = pd.concat(test_data) if test_data else pd.DataFrame(columns=train_df.columns)

print(f"Train set size: {len(train_df)} interactions")
print(f"Test set size: {len(test_df)} interactions")
print(f"Train ratio: {len(train_df) / (len(train_df) + len(test_df)):.2f}")

# Create train and test matrices
train_matrix = coo_matrix(
    (train_df['Quantity'], (train_df['user_idx'], train_df['item_idx'])),
    shape=(n_users, n_items)
)

test_matrix = coo_matrix(
    (test_df['Quantity'], (test_df['user_idx'], test_df['item_idx'])),
    shape=(n_users, n_items)
)

# Make test matrix binary (1 for interaction, regardless of quantity)
test_matrix.data = np.ones_like(test_matrix.data)

# Hyperparameters for the model
num_components = 64  # Higher dimensionality for latent factors
learning_rate = 0.05
epochs = 30
regularization = 0.001

# Train model with improved parameters
print("Training model on training set...")
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
    epochs=epochs, 
    num_threads=4,
    verbose=True
)

# Evaluate model
print("Evaluating model on test set...")
precision = precision_at_k(
    model, 
    test_matrix, 
    user_features=user_features,
    item_features=item_features,
    k=5
).mean()

recall = recall_at_k(
    model, 
    test_matrix, 
    user_features=user_features,
    item_features=item_features,
    k=5
).mean()

f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Test set evaluation results:")
print(f"Precision@5: {precision:.4f}")
print(f"Recall@5: {recall:.4f}")
print(f"F1@5: {f1:.4f}")


# Save model and encoders
print("Saving model and related data...")
joblib.dump(model, 'final_model.pkl')
joblib.dump((user_encoder, item_encoder), 'encoders.pkl')
joblib.dump(item_features, 'item_features.pkl')
joblib.dump(user_features, 'user_features.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

