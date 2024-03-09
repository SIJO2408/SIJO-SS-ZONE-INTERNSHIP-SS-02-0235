import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Load the MovieLens dataset (replace with your dataset)
# You can download the MovieLens dataset from: https://grouplens.org/datasets/movielens/
movies = pd.read_csv(r'C:\Users\SIJO BIJU\Downloads\archive\movies.csv')  # Replace 'movies.csv' with your movie dataset
ratings = pd.read_csv(r'C:\Users\SIJO BIJU\Downloads\archive\ratings.csv')  # Replace 'ratings.csv' with your ratings dataset

# Merge movies and ratings data
movie_ratings = pd.merge(ratings, movies, on='movieId')

# Create a user-item matrix
user_movie_ratings = movie_ratings.pivot_table(index='userId', columns='title', values='rating')

# Fill NaN values with 0
user_movie_ratings = user_movie_ratings.fillna(0)

# Transpose the matrix to have movies as rows and users as columns
movie_user_ratings = user_movie_ratings.T

# Calculate cosine similarity between users
user_similarity = cosine_similarity(movie_user_ratings)

# Convert the similarity matrix to a DataFrame
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_ratings.columns, columns=user_movie_ratings.columns)

# Function to get movie recommendations for a user
def get_movie_recommendations(user_ratings):
    similar_scores = user_similarity_df[user_ratings.index].sum(axis=1)
    similar_users = similar_scores.sort_values(ascending=False).index
    recommendations = []
    
    for movie in user_movie_ratings.index:
        if user_ratings[movie] == 0:
            movie_score = (user_movie_ratings[movie] * user_similarity_df.loc[movie][similar_users]).sum()
            recommendations.append((movie, movie_score))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:10]

# Example: Provide movie recommendations for a specific user (replace with your own user preferences)
user_preferences = {
    'Inception': 5.0,
    'The Dark Knight': 4.5,
    'Toy Story': 3.0,
    'Pulp Fiction': 4.0
}

user_ratings = pd.Series(user_preferences)

# Get movie recommendations
recommendations = get_movie_recommendations(user_ratings)

# Display the recommendations
print("Top 10 Movie Recommendations:")
for movie, score in recommendations:
    print(f"{movie} - Score: {score}")
    
