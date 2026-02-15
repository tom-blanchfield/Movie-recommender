import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Recommender Test Rig", layout="wide")
st.title("🎯 Recommender Test Rig")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    return ratings, movies

ratings, movies = load_data()

# ---------- SIDEBAR CONTROLS ----------
st.sidebar.header("Test Rig Parameters")

num_test_users = st.sidebar.slider("Number of test users", min_value=5, max_value=50, value=20, step=5)
num_ratings_used = st.sidebar.slider("Ratings per test user", min_value=5, max_value=50, value=10, step=5)
num_neighbors = st.sidebar.slider("Number of cosine neighbors", min_value=1, max_value=20, value=5, step=1)
top_n_precision = st.sidebar.slider("Top N for Precision@N", min_value=1, max_value=20, value=5)

st.sidebar.markdown("---")
st.sidebar.markdown("Weights: Weighted average uses similarity scores to weight neighbors' ratings.")

# ---------- PREPARE USER-MOVIE MATRIX ----------
user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

all_user_ids = user_movie_matrix.index.tolist()
all_movie_ids = user_movie_matrix.columns.tolist()

# ---------- SELECT RANDOM TEST USERS ----------
np.random.seed(42)
test_user_ids = np.random.choice(all_user_ids, size=num_test_users, replace=False)

# ---------- RESULTS ----------
rmse_list = []
precision_list = []

for uid in test_user_ids:
    user_ratings = user_movie_matrix.loc[uid]
    rated_movies = user_ratings[user_ratings > 0].index.tolist()

    if len(rated_movies) <= num_ratings_used:
        train_movies = rated_movies[:-1]  # leave at least 1 for testing
    else:
        train_movies = np.random.choice(rated_movies, size=num_ratings_used, replace=False).tolist()

    test_movies = [m for m in rated_movies if m not in train_movies]

    # Build user vector for train movies
    user_vector = np.zeros(len(all_movie_ids))
    movie_id_to_index = {m: i for i, m in enumerate(all_movie_ids)}

    for m in train_movies:
        user_vector[movie_id_to_index[m]] = user_movie_matrix.loc[uid, m]

    # Compute similarities with other users
    similarities = cosine_similarity([user_vector], user_movie_matrix.values)[0]

    # Exclude the user themselves
    similarities[all_user_ids.index(uid)] = -1

    # Find top neighbors
    neighbor_indices = np.argsort(similarities)[-num_neighbors:]
    neighbor_sims = similarities[neighbor_indices]
    neighbor_ids = user_movie_matrix.index[neighbor_indices]

    # Predict ratings for test movies
    preds = []
    actuals = []

    for m in test_movies:
        neighbor_ratings = []
        weights = []

        for n_idx, n_id in zip(neighbor_indices, neighbor_ids):
            rating = user_movie_matrix.loc[n_id, m]
            if rating > 0:
                neighbor_ratings.append(rating)
                weights.append(similarities[n_idx])

        if neighbor_ratings:
            # Weighted average
            pred = np.dot(neighbor_ratings, weights) / np.sum(weights)
            preds.append(pred)
            actuals.append(user_movie_matrix.loc[uid, m])

    if preds:
        # RMSE
        rmse = mean_squared_error(actuals, preds, squared=False)
        rmse_list.append(rmse)

        # Precision@N
        top_pred_idx = np.argsort(preds)[-top_n_precision:]
        top_actual_idx = np.argsort(actuals)[-top_n_precision:]
        # Compute number of overlaps
        precision = len(set(top_pred_idx).intersection(set(top_actual_idx))) / top_n_precision
        precision_list.append(precision)

# ---------- DISPLAY RESULTS ----------
st.subheader("Test Rig Results")
st.write(f"Tested {num_test_users} users using {num_ratings_used} ratings each and {num_neighbors} neighbors.")

if rmse_list:
    st.write(f"✅ Average RMSE: {np.mean(rmse_list):.3f}")
    st.write(f"✅ Median RMSE: {np.median(rmse_list):.3f}")
else:
    st.write("⚠️ No predictions could be made with current settings.")

if precision_list:
    st.write(f"✅ Average Precision@{top_n_precision}: {np.mean(precision_list):.3f}")
    st.write(f"✅ Median Precision@{top_n_precision}: {np.median(precision_list):.3f}")
