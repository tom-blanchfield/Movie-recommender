import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

st.set_page_config(page_title="Recommender Test Rig", layout="wide")
st.title("🧪 Movie Recommender Test Rig")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    return ratings, movies

ratings, movies = load_data()

# -------------------- PARAMETERS --------------------
st.sidebar.header("Test Parameters")

num_test_users = st.sidebar.slider("Number of test users", 5, 50, 20)
ratings_used = st.sidebar.slider("Ratings per test user", 5, 30, 10)
num_neighbors = st.sidebar.slider("Cosine neighbours", 3, 30, 5)
min_overlap = st.sidebar.slider("Minimum overlap", 1, 20, 5)

# -------------------- USER MATRIX --------------------
user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

user_means = user_movie_matrix.mean(axis=1)
mean_centered = user_movie_matrix.sub(user_means, axis=0).fillna(0)

all_users = user_movie_matrix.index.tolist()

# -------------------- RMSE FUNCTION --------------------
def rmse(actuals, preds):
    actuals = np.array(actuals)
    preds = np.array(preds)
    return np.sqrt(np.mean((actuals - preds) ** 2))

# -------------------- TEST BUTTON --------------------
if st.button("Run Test"):

    test_users = random.sample(all_users, num_test_users)
    user_rmses = []

    progress = st.progress(0)

    for idx, user_id in enumerate(test_users):

        user_row = user_movie_matrix.loc[user_id].dropna()

        if len(user_row) <= ratings_used + 1:
            continue

        known_ratings = user_row.sample(ratings_used)
        hidden_ratings = user_row.drop(known_ratings.index)

        # Build target vector
        target_vector = pd.Series(0, index=mean_centered.columns, dtype=float)
        target_vector[known_ratings.index] = known_ratings.values

        target_mean = known_ratings.mean()
        target_vector = target_vector - target_mean
        target_vector = target_vector.fillna(0)

        # Cosine similarity
        similarities = cosine_similarity([target_vector], mean_centered.values)[0]

        # Overlap filter
        overlaps = (mean_centered != 0).dot((target_vector != 0).astype(int))
        valid_users = np.where(overlaps >= min_overlap)[0]

        if len(valid_users) == 0:
            continue

        sims_filtered = similarities[valid_users]
        top_indices = valid_users[np.argsort(sims_filtered)[-num_neighbors:]]

        preds = []
        actuals = []

        for movie_id, actual_rating in hidden_ratings.items():

            num = 0
            den = 0

            for u_idx in top_indices:
                sim = similarities[u_idx]
                if sim <= 0:
                    continue

                rating = mean_centered.iloc[u_idx][movie_id]
                if rating != 0:
                    num += sim * rating
                    den += abs(sim)

            if den > 0:
                pred_rating = target_mean + (num / den)
                preds.append(pred_rating)
                actuals.append(actual_rating)

        if len(preds) > 0:
            user_rmses.append(rmse(actuals, preds))

        progress.progress((idx + 1) / len(test_users))

    if user_rmses:
        avg_rmse = np.mean(user_rmses)
        st.success(f"Average RMSE: {avg_rmse:.3f}")
        st.write(f"Users evaluated: {len(user_rmses)}")
    else:
        st.error("No valid users evaluated — try lowering overlap or ratings used.")
