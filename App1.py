import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity  # still used for optional genre/tag filtering

st.set_page_config(page_title="Movie Recommender 2.0 (NMF)", layout="wide")
st.title("🎬 MoRiS 2.0 — Movie Recommender (NMF)")

# ---------- SETTINGS ----------
TMDB_API_KEY = "888bb40cd1f4d3c95b375753e9c34c09"
MIN_REC_RATINGS = 10
MIN_OVERLAP = 5

# ---------- DATA PATHS ----------
RATINGS_PATH = "ratings.csv"
MOVIES_PATH = "movies.csv"
LINKS_PATH = "links.csv"
TAGS_PATH = "tags.csv"
NMF_MODEL_PATH = "nmf_model_200.npz"  # <-- pre-trained file in repo

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)
    tags = pd.read_csv(TAGS_PATH)
    links = pd.read_csv(LINKS_PATH)
    return ratings, movies, tags, links

ratings, movies, tags, links = load_data()

movies["genres"] = movies["genres"].fillna("")
movies = movies.merge(links, on="movieId", how="left")

# ---------- RATING STATS ----------
rating_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
rating_stats.columns = ["movieId", "avg_rating", "rating_count"]
movies = movies.merge(rating_stats, on="movieId", how="left").fillna({"avg_rating":0, "rating_count":0})

# ---------- POSTER FETCH ----------
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
            return f"https://image.tmdb.org/t/p/w400{poster_path}"
    except Exception:
        return None
    return None

# ---------- GENRES ----------
all_genres = sorted(
    set(
        g
        for sub in movies["genres"].str.split("|")
        for g in sub
        if g and g != "(no genres listed)"
    )
)

# ---------- SESSION STATE ----------
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

# =========================================================
# LOAD NMF MODEL
# =========================================================
@st.cache_data
def load_nmf_model(path):
    data = np.load(path, allow_pickle=True)
    W = data["W"]
    H = data["H"]
    user_ids = data["user_ids"]
    movie_ids = data["movie_ids"]
    pred_matrix = pd.DataFrame(np.dot(W, H), index=user_ids, columns=movie_ids)
    return pred_matrix

nmf_pred_matrix = load_nmf_model(NMF_MODEL_PATH)

# =========================================================
# DISCOVER PANEL
# =========================================================
st.subheader("Discover Movies")
mode = st.selectbox("Choose recommendation mode", ["Genres", "Keywords"])
genre_tag_movies = movies.copy()

if mode == "Genres":
    selected_genres = st.multiselect("Select Genres", all_genres)
    if selected_genres:
        genre_tag_movies["genre_score"] = genre_tag_movies["genres"].apply(
            lambda g: sum(1 for sel in selected_genres if sel in g)
        )
    else:
        genre_tag_movies["genre_score"] = 0
    genre_tag_movies["tag_score"] = 0

elif mode == "Keywords":
    selected_tags = st.multiselect("Keywords (press Enter after each)", options=[], default=[], accept_new_options=True)
    selected_tags = [t.lower() for t in selected_tags]
    genre_tag_movies["genre_score"] = 0
    if selected_tags:
        tag_mask = tags["tag"].str.lower().apply(lambda t: any(sel in t for sel in selected_tags))
        tag_filtered = tags[tag_mask]
        tag_counts = tag_filtered["movieId"].value_counts()
        genre_tag_movies["tag_score"] = genre_tag_movies["movieId"].map(tag_counts).fillna(0)
    else:
        genre_tag_movies["tag_score"] = 0

genre_tag_movies["total_score"] = genre_tag_movies["genre_score"] + genre_tag_movies["tag_score"]

if (mode == "Genres" and selected_genres) or (mode == "Keywords" and selected_tags):
    ranked_movies = genre_tag_movies[
        (genre_tag_movies["total_score"] > 0) &
        (genre_tag_movies["rating_count"] >= MIN_REC_RATINGS)
    ].sort_values(by=["avg_rating", "rating_count"], ascending=False)

    for _, row in ranked_movies.head(30).iterrows():
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
movie_search = st.text_input("Type part of a movie title")
if movie_search:
    filtered_titles = movies[movies["title"].str.contains(movie_search, case=False, na=False)]["title"].tolist()
else:
    filtered_titles = []

selected_movie = st.selectbox("Select movie", options=filtered_titles if filtered_titles else ["No results"])
rating_value = st.slider("Rating", 1, 5, 3)

if st.button("Add Rating") and filtered_titles:
    movie_id = int(movies[movies["title"] == selected_movie]["movieId"].values[0])
    st.session_state.user_ratings[movie_id] = int(rating_value)

if st.session_state.user_ratings:
    st.write("### Your Ratings")
    for m_id, r in st.session_state.user_ratings.items():
        title = movies[movies["movieId"] == m_id]["title"].values[0]
        st.write(f"{title}: {r}")

# =========================================================
# COLLAB RECOMMENDATIONS USING NMF
# =========================================================
if st.button("Get NMF Recommendations") and len(st.session_state.user_ratings) > 0:

    user_vec = pd.Series(0, index=nmf_pred_matrix.columns)
    for m_id, r in st.session_state.user_ratings.items():
        if m_id in user_vec.index:
            user_vec[m_id] = r

    # Optionally mean-center ratings (or use as-is)
    user_mean = user_vec[user_vec > 0].mean() if (user_vec > 0).any() else 0
    user_vec_centered = (user_vec - user_mean).fillna(0)

    # Simple prediction: dot with H if user is new
    preds = np.dot(user_vec_centered.values, nmf_pred_matrix.values.T).flatten() if "user_vec_name" not in nmf_pred_matrix.index else nmf_pred_matrix.loc[user_vec.name]
    preds_series = pd.Series(preds, index=nmf_pred_matrix.columns)
    
    # Remove already rated movies
    preds_series = preds_series.drop(index=[mid for mid in st.session_state.user_ratings.keys() if mid in preds_series.index])

    top_movies = preds_series.sort_values(ascending=False).head(30)

    st.subheader("Recommended Movies (NMF)")

    for m_id in top_movies.index:
        row = movies[movies["movieId"] == m_id].iloc[0]
        poster = get_poster(row["tmdbId"])
        if poster:
            st.markdown(
                f"<div style='text-align:center'>"
                f"<img src='{poster}' width='300'><br>"
                f"<strong>{row['title']}</strong>"
                f"</div>",
                unsafe_allow_html=True
            )
