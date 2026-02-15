import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import random

st.set_page_config(layout="wide")
st.title("Recommender Evaluation Lab")

@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv")
    return ratings

ratings = load_data()

# ---------- USER-MOVIE MATRIX ----------
user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

all_users = user_movie_matrix.index.tolist()

# ---------- CONTROLS ----------
num_test_users = st.slider("Number of Test Users", 5, 50, 10)
num_known_ratings = st.slider("Known Ratings per User", 5, 50, 10)
num_neighbors = st.slider("Number of Cosine Neighbours", 5, 50, 20)

# ---------- EVALUATION ----------
if st.button("Run Evaluation"):

    rmses = []

    test_users = random.sample(all_users, num_test_users)

    for test_user in test_users:

        user_ratings = ratings[ratings["userId"] == test_user]

        if len(user_ratings) <= num_known_ratings:
            continue

        user_ratings = user_ratings.sample(frac=1)

        train = user_ratings.head(num_known_ratings)
        test = user_ratings.tail(len(user_ratings) - num_known_ratings)

        # ---------- BUILD USER VECTOR ----------
        user_vector = np.zeros(user_movie_matrix.shape[1])
        movie_index_map = {m:i for i,m in enumerate(user_movie_matrix.columns)}

        for _, row in train.iterrows():
            if row["movieId"] in movie_index_map:
                user_vector[movie_index_map[row["movieId"]]] = row["rating"]

        # ---------- COSINE SIMILARITY ----------
        sims = cosine_similarity([user_vector], user_movie_matrix.values)[0]
        neighbor_ids = user_movie_matrix.index[np.argsort(sims)[-num_neighbors:]]

        preds = []
        actuals = []

        for _, row in test.iterrows():
            m_id = row["movieId"]
            true_rating = row["rating"]

            neighbor_ratings = ratings[
                (ratings["userId"].isin(neighbor_ids)) &
                (ratings["movieId"] == m_id)
            ]["rating"]

            if len(neighbor_ratings) == 0:
                continue

            pred_rating = neighbor_ratings.mean()

            preds.append(pred_rating)
            actuals.append(true_rating)

        if preds:
            mse = mean_squared_error(actuals, preds)
            rmse = np.sqrt(mse)

            rmses.append(rmse)

    if rmses:
        st.success(f"Average RMSE: {np.mean(rmses):.3f}")
        st.write("Individual RMSEs:", rmses)
    else:
        st.warning("Not enough data to evaluate.")
