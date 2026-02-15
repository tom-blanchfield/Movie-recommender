import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Recommender Test Rig", layout="wide")
st.title("🎛️ Movie Recommender Test Rig")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    return ratings, movies

ratings, movies = load_data()

# ---------- SIDEBAR CONTROLS ----------
st.sidebar.header("Test Rig Parameters")
num_test_users = st.sidebar.slider("Number of test users", min_value=5, max_value=50, value=20, step=1)
ratings_per_user = st.sidebar.slider("Number of known ratings per test user", min_value=1, max_value=20, value=10, step=1)
num_neighbors = st.sidebar.slider("Number of cosine neighbors", min_value=1, max_value=20, value=5, step=1)
top_n_precision = st.sidebar.slider("Top-N for precision calculation", min_value=1, max_value=20, value=10, step=1)
calculate_precision = st.sidebar.checkbox("Calculate Precision@N", value=True)

# ---------- PREP DATA ----------
# pivot user-movie matrix
user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

all_user_ids = ratings["userId"].unique()
test_user_ids = np.random.choice(all_user_ids, size=num_test_users, replace=False)

rmse_list = []
precision_list = []

# ---------- TEST LOOP ----------
for test_user in test_user_ids:
    user_ratings = ratings[ratings["userId"] == test_user]
    
    # Skip users with fewer ratings than needed
    if len(user_ratings) <= ratings_per_user:
        continue
    
    # Randomly select known ratings
    known_ratings = user_ratings.sample(n=ratings_per_user, random_state=42)
    test_ratings = user_ratings.drop(known_ratings.index)
    
    # Create test user vector
    user_vector = np.zeros(user_movie_matrix.shape[1])
    movie_id_to_index = {int(m): i for i, m in enumerate(user_movie_matrix.columns)}
    for _, row in known_ratings.iterrows():
        movie_id = int(row["movieId"])
        if movie_id in movie_id_to_index:
            user_vector[movie_id_to_index[movie_id]] = row["rating"]
    
    # Compute cosine similarity
    similarities = cosine_similarity([user_vector], user_movie_matrix.values)[0]
    similar_users_idx = np.argsort(similarities)[-num_neighbors:]
    similar_user_ids = user_movie_matrix.index[similar_users_idx]
    
    # Aggregate neighbor ratings (weighted by similarity)
    neighbor_ratings = ratings[ratings["userId"].isin(similar_user_ids)]
    predictions = []
    actuals = []
    
    for _, row in test_ratings.iterrows():
        movie_id = int(row["movieId"])
        actual_rating = row["rating"]
        # weighted average
        neighbors_for_movie = neighbor_ratings[neighbor_ratings["movieId"] == movie_id]
        if not neighbors_for_movie.empty:
            sim_scores = [similarities[list(user_movie_matrix.index).index(uid)] for uid in neighbors_for_movie["userId"]]
            weighted_pred = np.average(neighbors_for_movie["rating"], weights=sim_scores)
            predictions.append(weighted_pred)
            actuals.append(actual_rating)
    
    # RMSE
    if predictions:
        rmse = mean_squared_error(actuals, predictions, squared=False)
        rmse_list.append(rmse)
    
    # Precision@N
    if calculate_precision:
        top_pred = pd.Series(predictions, index=[row["movieId"] for _, row in test_ratings.iterrows()]).sort_values(ascending=False).head(top_n_precision)
        liked = test_ratings[test_ratings["rating"] >= 4]["movieId"]
        hits = sum(mid in liked.values for mid in top_pred.index)
        precision = hits / top_n_precision
        precision_list.append(precision)

# ---------- RESULTS ----------
st.subheader("Evaluation Results")
if rmse_list:
    st.metric("Average RMSE", f"{np.mean(rmse_list):.3f}")
else:
    st.write("Not enough data for selected parameters.")

if calculate_precision and precision_list:
    st.metric(f"Average Precision@{top_n_precision}", f"{np.mean(precision_list):.3f}")

# Optional: display full distribution plots
st.subheader("RMSE Distribution")
if rmse_list:
    st.bar_chart(pd.Series(rmse_list, name="RMSE"))

if calculate_precision and precision_list:
    st.subheader("Precision@N Distribution")
    st.bar_chart(pd.Series(precision_list, name=f"Precision@{top_n_precision}"))
