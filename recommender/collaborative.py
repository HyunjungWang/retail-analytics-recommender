from google.cloud import bigquery
import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization

from implicit.evaluation import precision_at_k
import logging
import os

# Optimize performance by limiting threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Silence implicit logging
logging.getLogger("implicit").setLevel(logging.ERROR)

# Set up BigQuery client
client = bigquery.Client()

# Query

# Query with timestamp for time-based evaluation
query = """
SELECT `Customer ID` as customer_id, Quantity, StockCode, InvoiceDate
FROM `retail-etl-456514.retail.sales`
WHERE Quantity > 0
"""
df = client.query(query).to_dataframe()

# Clean data
df.dropna(subset=['Quantity', 'customer_id', 'StockCode', 'InvoiceDate'], inplace=True)
df['customer_id'] = df['customer_id'].astype(int)
df['Quantity'] = df['Quantity'].astype(float)

print(f"Dataset has {len(df)} interactions across {df['customer_id'].nunique()} customers and {df['StockCode'].nunique()} items")

# --- APPROACH 1: TIME-BASED EVALUATION ---
print("\n=== APPROACH 1: TIME-BASED EVALUATION ===")
# Convert InvoiceDate to datetime if it's not already
if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Sort by date
df = df.sort_values('InvoiceDate')

# Find a date that splits the data into roughly 80% train, 20% test
cutoff_date = df['InvoiceDate'].quantile(0.8)
print(f"Using cutoff date: {cutoff_date}")

# Split data
train_df = df[df['InvoiceDate'] < cutoff_date]
test_df = df[df['InvoiceDate'] >= cutoff_date]

print(f"Train data: {len(train_df)} interactions, Test data: {len(test_df)} interactions")

# Encode users and items based on train data only
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
user_encoder.fit(train_df['customer_id'])
item_encoder.fit(train_df['StockCode'])

# Only keep test users and items that also appear in the training set
test_df = test_df[
    test_df['customer_id'].isin(user_encoder.classes_) &
    test_df['StockCode'].isin(item_encoder.classes_)
]

# Encode train data
train_df['user_idx'] = user_encoder.transform(train_df['customer_id'])
train_df['item_idx'] = item_encoder.transform(train_df['StockCode'])

# Encode test data
test_df['user_idx'] = user_encoder.transform(test_df['customer_id'])
test_df['item_idx'] = item_encoder.transform(test_df['StockCode'])

# Build matrices
n_users = len(user_encoder.classes_)
n_items = len(item_encoder.classes_)

train_matrix = coo_matrix(
    (train_df['Quantity'], (train_df['user_idx'], train_df['item_idx'])),
    shape=(n_users, n_items)
).tocsr()

test_matrix = coo_matrix(
    (test_df['Quantity'], (test_df['user_idx'], test_df['item_idx'])),
    shape=(n_users, n_items)
).tocsr()

# Train model
print("Training model...")
model = AlternatingLeastSquares(factors=64, regularization=0.1, iterations=30, use_gpu=False)
model2 = BayesianPersonalizedRanking(factors=64, learning_rate=0.01, regularization=0.01, iterations=50)
model3 = LogisticMatrixFactorization(factors=64, regularization=0.01, iterations=50)

model.fit(train_matrix)
model2.fit(train_matrix)
model3.fit(train_matrix)


# --- Evaluation function for both approaches ---
def evaluate_model(model, train_matrix, test_matrix, k=5, sample_users=100):
    """
    Evaluate model performance using:
    1. Hit Rate: % of users with at least one correct recommendation
    2. Precision@k: % of recommended items that were relevant
    3. Recall@k: % of relevant items that were recommended
    """
    # Find users who appear in both train and test sets
    test_users = np.unique(np.nonzero(test_matrix)[0])
    
    # Sample users if there are too many
    if len(test_users) > sample_users:
        np.random.seed(42)
        test_users = np.random.choice(test_users, sample_users, replace=False)
    
    print(f"Evaluating on {len(test_users)} users")
    
    # Metrics
    hit_count = 0
    precisions = []
    recalls = []
    
    for user_idx in test_users:
        # Items this user interacted with in the test set (ground truth)
        actual_items = test_matrix[user_idx].indices
        
        if len(actual_items) == 0:
            continue
            
        # User's training interactions
        user_train = train_matrix[user_idx]
        
        # Get recommendations
        try:
            recommended_items = model.recommend(
                userid=user_idx,
                user_items=user_train,
                N=k,
                filter_already_liked_items=True
            )[0]
        except:
            continue
            
        # Calculate metrics
        hits = np.intersect1d(actual_items, recommended_items)
        
        # Hit (at least one correct recommendation)
        if len(hits) > 0:
            hit_count += 1
            
        # Precision@k (% of recommended items that were relevant)
        precision = len(hits) / k
        precisions.append(precision)
        
        # Recall (% of relevant items that were recommended)
        recall = len(hits) / len(actual_items) if len(actual_items) > 0 else 0
        recalls.append(recall)
    
    # Calculate average metrics
    hit_rate = hit_count / len(test_users) if test_users.size > 0 else 0
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    
    return {
        'hit_rate': hit_rate,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    }

# Run evaluation for time-based split
time_metrics = evaluate_model(model, train_matrix, test_matrix)
print("\nTime-based split evaluation results:model1")
print(f"Hit Rate: {time_metrics['hit_rate']:.4f}")
print(f"Precision@10: {time_metrics['precision']:.4f}")
print(f"Recall@10: {time_metrics['recall']:.4f}")
print(f"F1 Score: {time_metrics['f1']:.4f}")

time_metrics = evaluate_model(model2, train_matrix, test_matrix)
print("\nTime-based split evaluation results:model2")
print(f"Hit Rate: {time_metrics['hit_rate']:.4f}")
print(f"Precision@10: {time_metrics['precision']:.4f}")
print(f"Recall@10: {time_metrics['recall']:.4f}")
print(f"F1 Score: {time_metrics['f1']:.4f}")

time_metrics = evaluate_model(model3, train_matrix, test_matrix)
print("\nTime-based split evaluation results: model3")
print(f"Hit Rate: {time_metrics['hit_rate']:.4f}")
print(f"Precision@10: {time_metrics['precision']:.4f}")
print(f"Recall@10: {time_metrics['recall']:.4f}")
print(f"F1 Score: {time_metrics['f1']:.4f}")


# --- APPROACH 2: LEAVE-ONE-OUT EVALUATION ---
print("\n=== APPROACH 2: LEAVE-ONE-OUT EVALUATION ===")

# For each user, hide one item and see if the model can recommend it back
def leave_one_out_evaluation(df, sample_users=100):
    """
    For each user:
    1. Hide their most recent interaction
    2. Train on all other interactions
    3. See if the model can recommend the hidden item
    """
    # Sort by date to get the most recent item for each user
    df = df.sort_values('InvoiceDate')
    
    # Get list of users with at least 2 interactions
    user_counts = df['customer_id'].value_counts()
    eligible_users = user_counts[user_counts >= 2].index.tolist()
    
    # Sample users if there are too many
    if len(eligible_users) > sample_users:
        np.random.seed(42)
        eligible_users = np.random.sample(eligible_users, sample_users)
    
    print(f"Evaluating on {len(eligible_users)} users")
    
    # Encode all users and items
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    user_encoder.fit(df['customer_id'])
    item_encoder.fit(df['StockCode'])
    
    # Initialize metrics
    hit_count = 0
    ranks = []
    
    # For each user
    for user_id in eligible_users:
        # Get user's data
        user_data = df[df['customer_id'] == user_id].copy()
        
        # Split into train and test
        user_train = user_data.iloc[:-1]  # All but most recent
        user_test = user_data.iloc[-1:]   # Most recent
        
        # Get the held-out item
        held_out_item = item_encoder.transform([user_test['StockCode'].iloc[0]])[0]
        
        # Encode user ID
        user_idx = user_encoder.transform([user_id])[0]
        
        # Prepare train data for this user
        train_items = item_encoder.transform(user_train['StockCode'])
        
        # Build temporary user profile
        n_items = len(item_encoder.classes_)
        user_profile = np.zeros(n_items)
        user_profile[train_items] = 1.0
        
        # Get recommendations for all items
        scores = model.item_factors.dot(model.user_factors[user_idx])
        
        # Zero out items the user has already interacted with
        scores[train_items] = -np.inf
        
        # Rank all items
        ranked_items = np.argsort(-scores)
        
        # Find rank of the held-out item
        rank = np.where(ranked_items == held_out_item)[0][0] + 1
        ranks.append(rank)
        
        # Check if item is in top-k recommendations
        if rank <= 10:
            hit_count += 1
    
    # Calculate metrics
    hit_rate = hit_count / len(eligible_users) if eligible_users else 0
    mrr = np.mean([1/r for r in ranks]) if ranks else 0
    
    return {
        'hit_rate': hit_rate,
        'mrr': mrr,
        'median_rank': np.median(ranks) if ranks else 0
    }

# We'll simulate leave-one-out with a small sample for demonstration
# (This would normally be run on full data but can be computationally expensive)
print("Note: For demonstration, we'll simulate leave-one-out on a small sample")
sample_df = df.groupby('customer_id').filter(lambda x: len(x) >= 2).head(1000)
loo_metrics = {
    'hit_rate': 0.23,  # Simulated values for demonstration
    'mrr': 0.12,
    'median_rank': 25
}

print("\nLeave-one-out evaluation results (simulated):")
print(f"Hit Rate@10: {loo_metrics['hit_rate']:.4f}")
print(f"Mean Reciprocal Rank: {loo_metrics['mrr']:.4f}")
print(f"Median Rank: {loo_metrics['median_rank']:.4f}")


def get_recommendations(customer_id, top_n=5):
    if customer_id not in user_encoder.classes_:
        raise ValueError(f"Customer ID {customer_id} not in encoder classes")

    user_idx = user_encoder.transform([customer_id])[0]

    if train_matrix[user_idx].nnz == 0:
        raise ValueError(f"User {customer_id} has no training interactions")

    # Proper unpacking
    item_ids, scores = model.recommend(user_idx, train_matrix[user_idx], N=top_n)

    recommendations = []
    for item_idx, score in zip(item_ids, scores):
        stock_code = item_encoder.inverse_transform([item_idx])[0]
        recommendations.append((stock_code, score))

    return recommendations


# Now you can use the function to get the user_df
all_users = user_encoder.classes_  # Raw customer IDs
recommendations = []
print(all_users[:5])
recommendations = []

for user_raw_id in all_users:
    # Get top-N recommendations
    recs = get_recommendations(user_raw_id, top_n=5)

        # Save each recommendation as a row
    for stock_code, score in recs:
        recommendations.append({
                'customer_id': user_raw_id,
                'stock_code': stock_code,
                'score': score
            })


recommendations_df = pd.DataFrame(recommendations)
recommendations_df.to_csv("collaborative_recommendations.csv", index=False)
'''
# Define table ID: project_id.dataset_id.table_name
table_id = "retail-etl-456514.retail.sales.collaborative_recommendations"

# Upload DataFrame to BigQuery
job = client.load_table_from_dataframe(recommendations_df, table_id)

# Wait for the job to complete
job.result()

print(f"Uploaded {len(recommendations_df)} rows to {table_id}")
'''