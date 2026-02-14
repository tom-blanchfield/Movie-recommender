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

# FIXED LINE (no underscore)
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

# =========================================================
# GENRE + TAG DISCOVERY
# =========================================================
st.divider()
st.subheader("Discover by Genre & Tags")

selected_genres = st.multiselect("Select Genres", all_genres)
tag_search = st.text_input("Optional Tag Keyword")

genre_tag_movies = movies.copy()

if selected_genres:
    genre_tag_movies["genre_score"] = genre_tag_movies["genres"].apply(
        lambda g: sum(1 for sel in selected_genres if sel in g)
    )
else:
    genre_tag_movies["genre_score"] = 0

if tag_search:
    tag_filtered = tags[
        tags["tag"].str.contains(tag_search, case=False, na=False)
    ]
    tag_counts = tag_filtered["movieId"].value_counts()
    genre_tag_movies["tag_score"] = genre_tag_movies["movieId"].map(tag_counts).fillna(0)
else:
    genre_tag_movies["tag_score"] = 0

genre_tag_movies["total_score"] = (
    genre_tag_movies["genre_score"] +
    genre_tag_movies["tag_score"]
)

if selected_genres or tag_search:
    ranked_movies = genre_tag_movies[
        genre_tag_movies["total_score"] > 0
    ].sort_values(by="total_score", ascending=False)

    st.write("### Matching Movies")
    for _, row in ranked_movies.head(25).iterrows():
        st.write(f"• {row['title']} (Score: {int(row['total_score'])})")

# =========================================================
# COLLABORATIVE RECOMMENDATIONS
# =========================================================
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

    movie_scores = movie_scores[
        ~movie_scores.index.isin(st.session_state.user_ratings.keys())
    ]

    movie_scores = movie_scores.head(20)

    rec_movies = movies.set_index("movieId").loc[movie_scores.index]

    st.subheader("Recommended Movies")
    for
