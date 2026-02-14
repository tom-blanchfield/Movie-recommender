import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Collaborative Movie Recommender")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    tags = pd.read_csv("tags.csv")
    return ratings, movies, tags

ratings, movies, tags = load_data()

# ---------- PREP GENRES ----------
movies["genres"] = movies["genres"].fillna("")
all_genres = sorted(
    set(
        g
        for sub in movies["genres"].str.split("|")
        for g in sub
        if g and g != "(no genres listed)"
    )
)

# ---------- TOP 50% MOST ACTIVE USERS ----------
user_counts = ratings["userId"].value_counts()
top_users = user_counts.head(int(len(user_counts) * 0.50)).index
ratings_top = ratings[ratings["userId"].isin(top_users)]

user_movie_matrix = ratings_top.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

# ---------- SESSION STATE ----------
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

# =========================================================
# RATE MOVIES
# =========================================================
st.subheader("Rate Movies")

movie_search = st.text_input("Search movie title")

filtered_movies = movies[
    movies["title"].str.contains(movie_search, case=False, na=False)
].head(15)

selected_movie = st.selectbox(
    "Select movie",
    filtered_movies["title"] if not filtered_movies.empty else ["No results"]
)

rating_value = st.slider("Your rating", 1, 5, 3)_
