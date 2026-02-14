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

# -------- TOP 10% MOST ACTIVE USERS --------
user_counts = ratings["userId"].value_counts()
top_users = user_counts.head(int(len(user_counts) * 0.50)).index
ratings_top = ratings[ratings["userId"].isin(top_users)]

# Pivot matrix
user_movie_matrix = ratings_top.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

# -------- SESSION STATE --------
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

st.subheader("Rate Movies")

movie_search = st.text_input("Search movie")

filtered_movies = movies[movies["title"].str.contains(movie_search, case=False, na=False)].head(10)

selected_movie = st.selectbox(
    "Select movie",
    filtered_movies["title"] if not filtered_movies.empty else ["No results"]
)

rating_value = st.slider("Your rating", 1, 5, 3)

if st.button("Add Rating"):
    movie_id = movies[movies["title"] == selected_movie]["movieId"].values[0]
    st.session_state.user_ratings[movie_id] = rating_value

st.write("Your ratings:", st.session_state.user_ratings)

# -------- RECOMMENDATIONS --------
if st.button("Get Recommendations") and len(st.session_state.user_ratings) > 0:

    # Build user vector
    user_vector = np.zeros(user_movie_matrix.shape[1])
    movie_id_to_index = {m:i for i,m in enumerate(user_movie_matrix.columns)}

    for m_id, r in st.session_state.user_ratings.items():
        if m_id in movie_id_to_index:
            user_vector[movie_id_to_index[m_id]] = r

    # Similarity
    similarities = cosine_similarity(
        [user_vector],
        user_movie_matrix.values
    )[0]

    similar_users_idx = np.argsort(similarities)[-10:]
    similar_user_ids = user_movie_matrix.index[similar_users_idx]

    # Collect favourite movies
    fav_movies = ratings_top[
        (ratings_top["userId"].isin(similar_user_ids)) &
        (ratings_top["rating"] >= 4)
    ]

    movie_scores = fav_movies["movieId"].value_counts().head(20)
    rec_movies = movies[movies["movieId"].isin(movie_scores.index)]

    st.subheader("Recommended Movies")
    for title in rec_movies["title"].values:
        st.write("•", title)
