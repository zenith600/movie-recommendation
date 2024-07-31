import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import pickle

# Load datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets
credits_renamed = credits.rename(columns={'movie_id': 'id'})
movies = movies.merge(credits_renamed, on='id')

# Resolve column conflicts by renaming or dropping unnecessary columns
movies = movies.rename(columns={'title_x': 'title'}).drop('title_y', axis=1)

# Preprocess data
movies['title'] = movies['title'].fillna('')
movies['cast'] = movies['cast'].apply(lambda x: ' '.join([i['name'] for i in eval(x)]) if isinstance(x, str) else '')
movies['crew'] = movies['crew'].apply(lambda x: ' '.join([i['name'] for i in eval(x)]) if isinstance(x, str) else '')
movies['keywords'] = movies['keywords'].apply(lambda x: ' '.join([i['name'] for i in eval(x)]) if isinstance(x, str) else '')
movies['genres'] = movies['genres'].apply(lambda x: ' '.join([i['name'] for i in eval(x)]) if isinstance(x, str) else '')

movies['combined'] = movies['title'] + ' ' + movies['cast'] + ' ' + movies['crew'] + ' ' + movies['keywords'] + ' ' + movies['genres']

# Content-based filtering
cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(movies['combined'])
content_sim = cosine_similarity(count_matrix, count_matrix)

# Similarity Model
similarity_model = {
    'movies': movies,
    'content_sim': content_sim,
    'cv': cv
}

# Save the similarity model
with open('similarity_movie_recommendation_model.pkl', 'wb') as file:
    pickle.dump(similarity_model, file)

print("Similarity Model saved successfully.")

# Collaborative filtering
ratings_data = movies[['id', 'vote_average', 'vote_count']]
ratings_data = ratings_data[ratings_data['vote_count'] > 50]  # Filter out movies with very few ratings

# Create a user-item matrix
user_item_matrix = pd.pivot_table(ratings_data, values='vote_average', index='id', columns='vote_count')
user_item_matrix = user_item_matrix.fillna(0)

# Perform matrix factorization
svd = TruncatedSVD(n_components=20, random_state=42)
latent_matrix = svd.fit_transform(user_item_matrix)

# Calculate similarity between movies based on their latent features
collaborative_sim = cosine_similarity(latent_matrix)

# Ensure both similarity matrices have the same shape
min_shape = min(content_sim.shape[0], collaborative_sim.shape[0])
content_sim = content_sim[:min_shape, :min_shape]
collaborative_sim = collaborative_sim[:min_shape, :min_shape]

# Hybrid model: Combine content-based and collaborative filtering results
hybrid_sim = content_sim * 0.5 + collaborative_sim * 0.5

# Hybrid Model
hybrid_model = {
    'movies': movies,
    'content_sim': content_sim,
    'collaborative_sim': collaborative_sim,
    'hybrid_sim': hybrid_sim,
    'cv': cv
}

# Save the hybrid model
with open('hybrid_movie_recommendation_model.pkl', 'wb') as file:
    pickle.dump(hybrid_model, file)

print("Hybrid Model saved successfully.")