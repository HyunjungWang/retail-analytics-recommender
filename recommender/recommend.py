import numpy as np
import joblib
import pandas as pd

# Load saved objects
model = joblib.load('model.pkl')
user_encoder, item_encoder = joblib.load('encoders.pkl')
user_features, item_features = joblib.load('features.pkl')

def recommend(user_raw_id, num_recommendations=5):
    # Transform raw user ID to encoded user ID
    user_id = user_encoder.transform([user_raw_id])[0]

    # Generate prediction scores for all items for the given user
    scores = model.predict(
        np.repeat(user_id, item_features.shape[0]),
        np.arange(item_features.shape[0]),
        item_features=item_features
    )
    
    # Get top recommended item indices
    top_items_idx = scores.argsort()[-num_recommendations:][::-1]

    # Convert indices back to raw item IDs
    return item_encoder.inverse_transform(top_items_idx)

# Generate recommendations for all users
from hybrid import get_user_df  # Import the function from hybrid.py

# Now you can use the function to get the user_df
user_df = get_user_df()
all_users = user_encoder.classes_  # This gives the encoded user IDs

recommendations = []

for user_raw_id in all_users:
    # Get the top recommended items for each user
    recommended_items = recommend(user_raw_id, num_recommendations=5)
    
    # Fetch the country information for this user from user_df (using encoded user ID)
    user_data = user_df[user_df['customer_id'] == user_raw_id]
    
    if not user_data.empty:
        recommendations.append([user_raw_id, *recommended_items])

# Convert the list of recommendations into a DataFrame
recommendation_df = pd.DataFrame(recommendations, columns=['user_id', 'rec_item_1', 'rec_item_2', 'rec_item_3', 'rec_item_4', 'rec_item_5'])

# Save recommendations to CSV
#recommendation_df.to_csv('recommendations_with_segments.csv', index=False)

print("Recommendations saved to 'recommendations_with_segments.csv'")
