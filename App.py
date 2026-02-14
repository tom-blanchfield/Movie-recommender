import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender")

# -------------------- TMDB API KEY --------------------
TMDB_API_KEY = "888bb40cd1f4d3c95b375753e9c34c09"

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    tags = pd.read_csv("tags.csv")
    links = pd.read_csv("links.csv")
    return ratings, movies, tags, links

ratings, movies, tags, links = load_data()

movies["genres"] = movies["genres"].fillna("")
movies = movies.merge(links, on="movieId", how="left")

# ---------- RATING STATS ----------
rating_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
rating_stats.columns = ["movieId", "avg_rating", "rating_count"]

movies = movies.merge(rating_stats, on="movieId", how="left")
movies["avg_rating"] = movies["avg_rating"].fillna(0)
movies["rating_count"] = movies["rating_count"].fillna(0)

MIN_RATINGS = 20

# ---------- POSTER FETCH FUNCTION ----------
@st.cache_data(show_spinner=False)
def get_poster(tmdb_id):
    if pd.isna(tmdb_id):
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}"
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        data = r.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w400{poster_path}"  # larger poster
    except Exception:
        return None
    return None

# ---------- GENRE LIST ----------
all_genres = sorted(
    set(
        g
        for sub in movies["genres"].str.split("|")
        for g in sub
        if g and g != "(no genres listed)"
    )
)

# ---------- TOP 50% USERS ----------
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
# DISCOVER MOVIES
# =========================================================
st.subheader("Discover Movies")

mode = st.selectbox("Choose recommendation mode", ["Genres", "Keywords"])

genre_tag_movies = movies.copy()

if mode == "Genres":
    selected_genres = st.multiselect("Select Genres", all_genres)
    # --- Genre Score ---
    if selected_genres:
        genre_tag_movies["genre_score"] = genre_tag_movies["genres"].apply(
            lambda g: sum(1 for sel in selected_genres if sel in g)
        )
    else:
        genre_tag_movies["genre_score"] = 0
    # No tag score in genre mode
    genre_tag_movies["tag_score"] = 0

elif mode == "Keywords":
    selected_tags = st.multiselect(
        "Keywords (press Enter after each)", options=[], default=[], accept_new_options=True
    )
    selected_tags = [t.lower() for t in selected_tags]
    genre_tag_movies["genre_score"] = 0
    if selected_tags:
        tag_mask = tags["tag"].str.lower().apply(lambda t: any(sel in t for sel in selected_tags))
        tag_filtered = tags[tag_mask]
        tag_counts = tag_filtered["movieId"].value_counts()
        genre_tag_movies["tag_score"] = genre_tag_movies["movieId"].map(tag_counts).fillna(0)
    else:
        genre_tag_movies["tag_score"] = 0

# --- Total Score ---
genre_tag_movies["total_score"] = genre_tag_movies["genre_score"] + genre_tag_movies["tag_score"]

# --- Filter and Rank ---
if (mode == "Genres" and selected_genres) or (mode == "Keywords" and selected_tags):
    ranked_movies = genre_tag_movies[
        (genre_tag_movies["total_score"] > 0) &
        (genre_tag_movies["rating_count"] >= MIN_RATINGS)
    ].sort_values(by=["avg_rating", "rating_count"], ascending=False)

    for _, row in ranked_movies.head(15).iterrows():
        poster = get_poster(row["tmdbId"])
        if poster:
            st.markdown(
                f"<div style='text-align:center'>"
                f"<img src='{poster}' width='300'><br>"
                f"<strong>{row['title']}</strong><br>"
                f"⭐ {row['avg_rating']:.2f} ({int(row['rating_count'])} ratings)"
                f"</div>",
                unsafe_allow_html=True
            )

st.divider()

# =========================================================
# RATE MOVIES
# =========================================================
st.subheader("Rate Movies")

movie_search = st.text_input("Search movie")

filtered_movies = movies[
    movies["title"].str.contains(movie_search, case=False, na=False)
].head(15)

selected_movie = st.selectbox(
    "Select movie",
    filtered_movies["title"] if not filtered_movies.empty else ["No results"]
)

rating_value = st.slider("Rating", 1, 5, 3)

if st.button("Add Rating") and not filtered_movies.empty:
    movie_id = int(
        movies[movies["title"] == selected_movie]["movieId"].values[0]
    )
    st.session_state.user_ratings[movie_id] = int(rating_value)

# Display user ratings
if st.session_state.user_ratings:
    st.write("### Your Ratings")
    for m_id, r in st.session_state.user_ratings.items():
        title = movies[movies["movieId"] == m_id]["title"].values[0]
        st.write(f"{title}: {r}")

# =========================================================
# COLLAB RECOMMENDATIONS
# =========================================================
if st.button("Get Recommendations") and len(st.session_state.user_ratings) > 0:

    user_vector = np.zeros(user_movie_matrix.shape[1])
    movie_id_to_index = {int(m): i for i, m in enumerate(user_movie_matrix.columns)}

    for m_id, r in st.session_state.user_ratings.items():
        if m_id in movie_id_to_index:
            user_vector[movie_id_to_index[m_id]] = r

    similarities = cosine_similarity([user_vector], user_movie_matrix.values)[0]
    similar_users_idx = np.argsort(similarities)[-10:]
    similar_user_ids = user_movie_matrix.index[similar_users_idx]

    fav_movies = ratings_top[
        (ratings_top["userId"].isin(similar_user_ids)) &
        (ratings_top["rating"] >= 4)
    ]

    movie_scores = fav_movies["movieId"].value_counts()
    movie_scores = movie_scores[
        ~movie_scores.index.isin(st.session_state.user_ratings.keys())
    ].head(12)

    st.subheader("Recommended Movies")

    rec_movies = [movies[movies["movieId"] == m_id].iloc[0] for m_id in movie_scores.index]

    for row in rec_movies:
        poster = get_poster(row["tmdbId"])
        if poster:
            st.markdown(
                f"<div style='text-align:center'>"
                f"<img src='{poster}' width='300'><br>"
                f"<strong>{row['title']}</strong>"
                f"</div>",
                unsafe_allow_html=True
            )
