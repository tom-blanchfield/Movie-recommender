import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Collaborative Movie Recommender")

@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    return ratings, movies

ratings, movies = load_data()

# ---------- TOP 10% MOST ACTIVE USERS ----------
user_counts = ratings["userId"].value_counts()
top_users = user_counts.head(int(len(user_counts) * 0.10)).index
ratings_top = ratings[ratings["userId"].isin(top_users)]

user_movie_matrix = ratings_top.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

# ---------- SESSION STATE ----------
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

st.subheader("Rate Movies")

search = st.text_input("Search movie title")

matches = movies[movies["title"].str.contains(search, case=False, na=False)].head(10)

if not matches.empty:
    selected_title = matches.iloc[0]["title"]
    st.write("Selected:", selected_title)

    rating_val = st.slider("Your rating", 1, 5, 3)

    if st.button("Add Rating"):
        movie_id = int(matches.iloc[0]["movieId"])  # FIX numpy int
        st.session_state.user_ratings[movie_id] = int(rating_val)

# Display rated movies nicely
if st.session_state.user_ratings:
    st.write("### Your Ratings")
    for m_id, r in st.session_state.user_ratings.items():
        title = movies[movies["movieId"] == m_id]["title"].values[0]
        st.write(f"{title}: {r}")

# ---------- RECOMMEND ----------
if st.button("Get Recommendations") and st.session_state.user_ratings:

    user_vector = np.zeros(user_movie_matrix.shape[1])
    movie_id_to_index = {int(m): i for i, m in enumerate(user_movie_matrix.columns)}

    for m_id, r in st.session_state.user_ratings.items():
        if m_id in movie_id_to_index:
            user_vector[movie_id_to_index[m_id]] = r

    similarities = cosine_similarity([user_vector], user_movie_matrix.values)[0]
    similar_users_idx = np.argsort(similarities)[-10:]
    similar_user_ids = user_movie_matrix.index[similar_users_idx]

    # ---------- STRICT 5 STAR AGREEMENT ----------
    sim_ratings = ratings_top[ratings_top["userId"].isin(similar_user_ids)]

    grouped = sim_ratings.groupby("movieId")["rating"]

    unanimous_5 = grouped.apply(lambda x: (x == 5).all())
    unanimous_ids = unanimous_5[unanimous_5].index

    # Count ratings
    rating_counts = sim_ratings[sim_ratings["movieId"].isin(unanimous_ids)] \
        .groupby("movieId").size().sort_values(ascending=False)

    # Remove movies user already rated
    rating_counts = rating_counts[~rating_counts.index.isin(st.session_state.user_ratings.keys())]

    top_ids = rating_counts.head(20).index
    rec_movies = movies[movies["movieId"].isin(top_ids)]

    st.subheader("Recommended Movies")
    for title in rec_movies["title"].values:
        st.write("•", title)
