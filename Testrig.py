import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import random
import os

st.set_page_config(layout="wide")
st.title("Recommender Evaluation Lab (App Algorithm vs. Baseline)")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    file_path = "ratings.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path} in {os.getcwd()}")
    
    ratings = pd.read_csv(file_path)
    if ratings.empty:
        raise ValueError(f"{file_path} is empty!")
    return ratings

ratings = load_data()

# ---------- USER MATRICES ----------
user_movie_matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")
user_means = user_movie_matrix.mean(axis=1)
mean_centered = user_movie_matrix.sub(user_means, axis=0).fillna(0)

all_users = user_movie_matrix.index.tolist()

# ---------- CONTROLS ----------
num_test_users = st.slider("Number of Test Users", 5, 50, 10)
num_known_ratings = st.slider("Known Ratings per User", 5, 50, 10)
num_neighbors = st.slider("Number of Cosine Neighbours", 1, 20, 5)
min_overlap = st.slider("Minimum Overlap for Neighbors", 1, 20, 5)

# ---------- EVALUATION ----------
if st.button("Run Evaluation"):

    rmses_app = []
    rmses_baseline = []

    test_users = random.sample(all_users, num_test_users)

    for test_user in test_users:
        user_ratings = ratings[ratings["userId"] == test_user]

        if len(user_ratings) <= num_known_ratings:
            continue

        user_ratings = user_ratings.sample(frac=1)
        train = user_ratings.head(num_known_ratings)
        test = user_ratings.tail(len(user_ratings) - num_known_ratings)

        # ---------- BUILD TARGET VECTOR ----------
        user_vector = pd.Series(0, index=mean_centered.columns, dtype=float)
        for _, row in train.iterrows():
            if row["movieId"] in user_vector.index:
                user_vector[row["movieId"]] = row["rating"]

        target_mean = user_vector[user_vector > 0].mean()
        user_vector_centered = (user_vector - target_mean).fillna(0)

        # ---------- COSINE SIMILARITY ----------
        sims = cosine_similarity([user_vector_centered], mean_centered.values)[0]

        # Overlap filter
        overlaps = (mean_centered != 0).dot((user_vector_centered != 0).astype(int))
        valid_users = np.where(overlaps >= min_overlap)[0]

        similarities_filtered = sims[valid_users]
        if len(similarities_filtered) == 0:
            continue

        top_idx = valid_users[np.argsort(similarities_filtered)[-num_neighbors:]]

        # ---------- APP ALGORITHM PREDICTIONS ----------
        preds_app = []
        actuals = []

        for _, row in test.iterrows():
            m_id = row["movieId"]
            true_rating = row["rating"]

            num = 0
            den = 0

            for idx in top_idx:
                sim = sims[idx]
                if sim <= 0:
                    continue
                rating = mean_centered.iloc[idx][m_id]
                if rating != 0:
                    num += sim * rating
                    den += abs(sim)

            if den > 0:
                pred = target_mean + (num / den)
                preds_app.append(pred)
                actuals.append(true_rating)

        if preds_app:
            rmse_app = np.sqrt(mean_squared_error(actuals, preds_app))
            rmses_app.append(rmse_app)

        # ---------- BASELINE PREDICTIONS (simple mean) ----------
        preds_base = []
        actuals_base = []

        for _, row in test.iterrows():
            m_id = row["movieId"]
            true_rating = row["rating"]

            neighbor_ratings = ratings[
                (ratings["userId"].isin(user_movie_matrix.index[top_idx])) &
                (ratings["movieId"] == m_id)
            ]["rating"]

            if len(neighbor_ratings) == 0:
                continue

            pred_base = neighbor_ratings.mean()
            preds_base.append(pred_base)
            actuals_base.append(true_rating)

        if preds_base:
            rmse_base = np.sqrt(mean_squared_error(actuals_base, preds_base))
            rmses_baseline.append(rmse_base)

    if rmses_app and rmses_baseline:
        st.success(f"Average RMSE (App Algorithm): {np.mean(rmses_app):.3f}")
        st.success(f"Average RMSE (Baseline Mean): {np.mean(rmses_baseline):.3f}")
        st.write("Individual RMSEs (App Algorithm):", rmses_app)
        st.write("Individual RMSEs (Baseline Mean):", rmses_baseline)
    else:
        st.warning("Not enough data to evaluate.")
