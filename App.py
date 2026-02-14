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
        if g != "(no genres listed)"
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

st.subheader("Rate Movies")

# ---------- SEARCH ----------
movie_search = st.text_input("Search movie title")

filtered_movies = movies[
    movies["title"].str.contains(movie_search, case=False, na=False)
].head(15)

selected_movie = st.selectbox(
    "Select movie",
    filtered_movies["title"] if not filtered_movies.empty else ["No results"]
)

rating_value = st.slider("Your rating", 1, 5, 3)

if st.button("Add Rating") and not filtered_movies.empty:
    movie_id = int(
        movies[movies["title"] == selected_movie]["movieId"].values[0]
    )
    st.session_state.user_ratings[movie_id] = int(rating_value)

# ---------- SHOW USER RATINGS ----------
if st.session_state.user_ratings:
    st.write("### Your Ratings")
    for m_id, r in st.session_state.user_ratings.items():
        title = movies[movies["movieId"] == m_id]["title"].values[0]
        st.write(f"{title}: {r}")

# ---------- GENRE + TAG DISCOVERY ----------
st.divider()
st.subheader("Discover by Genre or Tag")

col1, col2 = st.columns(2)

with col1:
    selected_genre = st.selectbox("Genre", ["Any"] + all_genres)

with col2:
    tag_search = st.text_input("Tag keyword (e.g. space, mafia, love)")

genre_tag_movies = movies.copy()

if selected_genre != "Any":
    genre_tag_movies = genre_tag_movies[
        genre_tag_movies["genres"].str.contains(selected_genre)
    ]

if tag_search:
    tag_movie_ids = tags[
        tags["tag"].str.contains(tag_search, case=False, na=False)
    ]["movieId"].unique()
    genre_tag_movies = genre_tag_movies[
        genre_tag_movies["movieId"].isin(tag_movie_ids)
    ]

if selected_genre != "Any" or tag_search:
    st.write("### Matching Movies")
    for title in genre_tag_movies["title"].head(20):
        st.write("•", title)

# ---------- RECOMMENDATIONS ----------
st.divider()

if st.button("Get Recommendations") and len(st.session_state.user_ratings) > 0:

    user_vector = np.zeros(user_movie_matrix.shape[1])

    movie_id_to_index = {
        int(m): i for i, m in enumerate(user_movie_matrix.columns)
    }

    for m_id, r in st.session_state.user_ratings.items():
        if m_id in movie_id_to_index:
            user_vector[movie_id_to_index[m_id]] = r

    similarities = cosine_similarity(
        [user_vector],
        user_movie_matrix.values
    )[0]

    similar_users_idx = np.argsort(similarities)[-10:]
    similar_user_ids = user_movie_matrix.index[similar_users_idx]

    fav_movies = ratings_top[
        (ratings_top["userId"].isin(similar_user_ids)) &
        (ratings_top["rating"] >= 4)
    ]

    movie_scores = fav_movies["movieId"].value_counts()

    # Remove already rated movies
    movie_scores = movie_scores[
        ~movie_scores.index.isin(st.session_state.user_ratings.keys())
    ]

    movie_scores = movie_scores.head(20)

    # Preserve ranking order
    rec_movies = movies.set_index("movieId").loc[movie_scores.index]

    st.subheader("Recommended Movies")
    for title in rec_movies["title"].values:
        st.write("•", title)
