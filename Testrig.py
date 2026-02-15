import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Recommender Test Rig", layout="wide")
st.title("🎯 Recommender Test Rig")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    try:
        ratings = pd.read_csv("ratings.csv")
        movies = pd.read_csv("movies.csv")
        return ratings, movies
    except Exception as e:
        st.error(f"Error loading CSV files: {e}")
        return None, None

ratings, movies = load_data()

if ratings is None or movies is None:
    st.stop()

# -------------------- USER CONTROLS --------------------
st.sidebar.header("Test Parameters")

num_test_users = st.sidebar.slider("Number of test users", 5, 50, 20)
ratings_per_user = st.sidebar.slider("Ratings used for prediction", 5, 50, 10)
num_neighbors = st.sidebar.slider("Cosine neighbors", 3, 30, 5)
min_movie_ratings = st.sidebar.slider("Min movie ratings", 5, 100, 20)

# -------------------- FILTER MOVIES --------------------
movie_counts = ratings["movieId"].value_counts()
valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
ratings = ratings[ratings["movieId"].isin(valid_movies)]

# -------------------- BUILD MATRIX --------------------
user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

# Mean-center
user_means = user_movie_matrix.mean(axis=1)
user_movie_centered = user_movie_matrix.sub(user_means, axis=0).fillna(0)

# -------------------- SELECT TEST USERS --------------------
all_users = user_movie_matrix.index.tolist()
np.random.shuffle(all_users)
test_users = all_users[:num_test_users]

st.write(f"Testing **{len(test_users)} users**")

# -------------------- TEST LOOP --------------------
rmse_results = []

for user_id in test_users:

    user_ratings = user_movie_matrix.loc[user_id].dropna()

    if len(user_ratings) < ratings_per_user + 1:
        continue

    known = user_ratings.sample(ratings_per_user, random_state=42)
    unknown = user_ratings.drop(known.index)

    # Build user vector
    user_vector = np.zeros(user_movie_centered.shape[1])
    movie_id_to_index = {
        m: i for i, m in enumerate(user_movie_centered.columns)
    }

    for m_id, r in known.items():
        if m_id in movie_id_to_index:
            idx = movie_id_to_index[m_id]
            user_vector[idx] = r - user_means[user_id]

    # Cosine similarity
    sims = cosine_similarity(
        [user_vector],
        user_movie_centered.values
    )[0]

    neighbor_idx = np.argsort(sims)[-num_neighbors:]
    neighbor_ids = user_movie_centered.index[neighbor_idx]
    neighbor_sims = sims[neighbor_idx]

    preds = []
    actuals = []

    for movie_id, actual_rating in unknown.items():

        if movie_id not in movie_id_to_index:
            continue

        m_idx = movie_id_to_index[movie_id]

        weighted_sum = 0
        sim_sum = 0

        for n_id, sim in zip(neighbor_ids, neighbor_sims):
            neighbor_rating = user_movie_centered.loc[n_id].iloc[m_idx]
            if neighbor_rating != 0:
                weighted_sum += sim * neighbor_rating
                sim_sum += abs(sim)

        if sim_sum == 0:
            continue

        pred_centered = weighted_sum / sim_sum
        pred = pred_centered + user_means[user_id]

        preds.append(pred)
        actuals.append(actual_rating)

    if len(preds) > 0:
        mse = np.mean((np.array(actuals) - np.array(preds)) ** 2)
        rmse = np.sqrt(mse)
        rmse_results.append(rmse)

# -------------------- RESULTS --------------------
st.divider()
st.subheader("Results")

if rmse_results:
    avg_rmse = np.mean(rmse_results)
    st.write(f"Average RMSE: **{avg_rmse:.4f}**")
    st.write(f"Users evaluated: **{len(rmse_results)}**")
else:
    st.write("Not enough data to evaluate.")
