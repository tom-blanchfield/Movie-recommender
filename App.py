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

rating_value = st.slider("Your rating", 1, 5, 3)

# Add rating button
add_rating_clicked = st.button("Add Rating")
if add_rating_clicked and not filtered_movies.empty:
    movie_id = int(
        movies[movies["title"] == selected_movie]["movieId"].values[0]
    )
    st.session_state.user_ratings[movie_id] = int(rating_value)

# Show user ratings
if st.session_state.user_ratings:
    st.write("### Your Ratings")
    for m_id, r in st.session_state.user_ratings.items():
        title = movies[movies["movieId"] == m_id]["title"].values[0]
        st.write(f"{title}: {r}")

# Move Get Recommendations button directly under Add Rating
get_rec_clicked = st.button("Get Recommendations")

# =========================================================
# GENRE + TAG DISCOVERY
# =========================================================
st.divider()
st.subheader("Discover by Genre & Tags")

# ---------- GENRES ----------
selected_genres = st.multiselect(
    "Select Genres",
    all_genres
)

# ---------- TAG CHIPS ----------
selected_tags = st.multiselect(
    "Add Keywords (press Enter after each)",
    options=[],
    default=[],
    accept_new_options=True
)

genre_tag_movies = movies.copy()

# GENRE SCORE
if selected_genres:
    genre_tag_movies["genre_score"] = genre_tag_movies["genres"].apply(
        lambda g: sum(1 for sel in selected_genres if sel in g)
    )
else:
    genre_tag_movies["genre_score"] = 0

# TAG SCORE
selected_tags = [t.lower() for t in selected_tags]
if selected_tags:
    tag_mask = tags["tag"].str.lower().apply(
        lambda t: any(sel in t for sel in selected_tags)
    )
    tag_filtered = tags[tag_mask_]()_
