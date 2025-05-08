import numpy as np
import joblib
import pandas as pd
from google.cloud import bigquery
# Set up BigQuery client
client = bigquery.Client()


model = joblib.load('model2.pkl')
user_encoder, item_encoder = joblib.load('encoders2.pkl')
user_features= joblib.load('user_features2.pkl')
item_features = joblib.load('item_features2.pkl')
vectorizer=joblib.load('vectorizer2.pkl')


# Function to get recommendations for a single user or multiple users
def get_recommendations(user_ids, model, user_encoder, item_encoder, item_features, user_features, n=10, batch_size=100):
    """
    Get top N recommendations for one or multiple users
    
    Parameters:
    -----------
    user_ids: single user ID, list of user IDs, or 'all' for all users
    model: trained LightFM model
    user_encoder: fitted LabelEncoder for users
    item_encoder: fitted LabelEncoder for items
    item_features: sparse matrix of item features
    user_features: sparse matrix of user features
    n: number of recommendations to return per user
    batch_size: number of users to process at once (to manage memory usage)
    
    Returns:
    --------
    Dictionary mapping user_ids to lists of (item_id, score) tuples
    """
    try:
        # Handle different input types for user_ids
        if user_ids == 'all':
            # Use all users in the encoder
            user_indices = np.arange(len(user_encoder.classes_))
            actual_user_ids = user_encoder.inverse_transform(user_indices)
        elif isinstance(user_ids, (list, np.ndarray)):
            # Process list of user IDs
            actual_user_ids = user_ids
            user_indices = user_encoder.transform(user_ids)
        else:
            # Process single user ID
            actual_user_ids = [user_ids]
            user_indices = [user_encoder.transform([user_ids])[0]]
        
        # Get all item indices
        all_item_indices = np.arange(len(item_encoder.classes_))
        n_items = len(all_item_indices)
        
        # Create a dictionary to store recommendations
        recommendations = {}
        
        # Process users in batches to manage memory
        for batch_start in range(0, len(user_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(user_indices))
            batch_user_indices = user_indices[batch_start:batch_end]
            batch_actual_ids = actual_user_ids[batch_start:batch_end]
            
            print(f"Processing recommendations for users {batch_start+1}-{batch_end} of {len(user_indices)}")
            
            # Generate predictions for this batch of users
            for i, (user_idx, user_id) in enumerate(zip(batch_user_indices, batch_actual_ids)):
                if (i + 1) % 10 == 0:  # Print progress every 10 users
                    print(f"  User {batch_start+i+1}/{len(user_indices)}")
                
                # Get scores for all items for this user
                scores = model.predict(
                    user_ids=[user_idx] * n_items,
                    item_ids=all_item_indices,
                    user_features=user_features,
                    item_features=item_features
                )
                
                # Get top N item indices for this user
                top_items_indices = np.argsort(-scores)[:n]
                
                # Convert back to original item IDs
                top_items = item_encoder.inverse_transform(top_items_indices)
                
                # Store results in the dictionary
                recommendations[user_id] = [(item, scores[idx]) for item, idx in zip(top_items, top_items_indices)]
        
        return recommendations
    
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return {}

# Method 1: Use the batch-enabled function to get all recommendations at once
def generate_all_recommendations(top_n=5):
    """
    Generate recommendations for all users using the batch approach
    """
    print("Generating recommendations for all users...")
    
    # Use the enhanced function with 'all' parameter
    all_recommendations = get_recommendations(
        'all', 
        model, 
        user_encoder, 
        item_encoder, 
        item_features, 
        user_features, 
        n=top_n,
        batch_size=500  # Adjust based on your memory constraints
    )
    
    # Convert the nested dictionary to a flat list of dictionaries
    recommendations_list = []
    for user_id, recs in all_recommendations.items():
        for rank, (stock_code, score) in enumerate(recs, 1):
            recommendations_list.append({
                'customer_id': user_id,
                'stock_code': stock_code,
                'score': score,
                'rank': rank
            })
    
    # Convert to DataFrame
    recommendations_df = pd.DataFrame(recommendations_list)
    print(f"Generated {len(recommendations_df)} recommendations for {len(all_recommendations)} users")
    
    return recommendations_df

    
# Generate recommendations using method 1 (faster for most cases)
df_recommendations = generate_all_recommendations(top_n=5)

# Optional: Save recommendations to CSV
df_recommendations.to_csv('hybrid_recommendations2.csv', index=False)
print("Recommendations saved to user_item_recommendations.csv")

# Wait for the job to complete
client = bigquery.Client(project="retail-etl-456514")
dataset_id = 'retail'
table_id = 'hybrid_recommendations'

# Define the BigQuery table reference
table_ref = client.dataset(dataset_id).table(table_id)

# Upload the DataFrame to BigQuery (create a new table)
job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("customer_id", "STRING"),
        bigquery.SchemaField("stock_code", "STRING"),
        bigquery.SchemaField("score", "FLOAT"),
        bigquery.SchemaField("rank", "INTEGER")


    ],
    write_disposition="WRITE_EMPTY",  # This ensures no merging, it creates a new table
)

# Load the DataFrame into BigQuery
job = client.load_table_from_dataframe(df_recommendations, table_ref, job_config=job_config)
job.result()  # Wait for the job to complete
print("Uploaded done")
