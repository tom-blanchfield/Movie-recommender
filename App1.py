import streamlit as st
import pandas as pd
import numpy as np
import requests
from numpy.linalg import pinv

st.set_page_config(page_title="Movie Recommender 2.0", layout="wide")
st.title("🎬 MoRiS 2.0 — Movie Recommender")

# -------------------- SETTINGS --------------------
TMDB_API_KEY = "YOUR_KEY"
RATINGS_PATH = "ratings.csv"
MOVIES_PATH = "movies.csv"
LINKS_PATH = "links.csv"
TAGS_PATH = "tags.csv"
NMF_MODEL_PATH = "nmf_small_200f.npz"

BATCH_SIZE = 30

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)
    links = pd.read_csv(LINKS_PATH)
    tags = pd.read_csv(TAGS_PATH)
    return ratings, movies, links, tags

ratings, movies, links, tags = load_data()

movies = movies.merge(links, on="movieId", how="left")
movies["genres"] = movies["genres"].fillna("")

rating_stats = ratings.groupby("movieId")["rating"].agg(["mean","count"]).reset_index()
rating_stats.columns = ["movieId","avg_rating","rating_count"]
movies = movies.merge(rating_stats, on="movieId", how="left")
movies.fillna({"avg_rating":0,"rating_count":0}, inplace=True)

# FAST LOOKUPS
movie_dict = movies.set_index("movieId").to_dict("index")
title_to_id = dict(zip(movies["title"], movies["movieId"]))

# -------------------- SESSION --------------------
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}
if "nmf_rec_index" not in st.session_state:
    st.session_state.nmf_rec_index = 0

# -------------------- POSTER --------------------
@st.cache_data(show_spinner=False)
def get_poster(tmdb_id):
    if pd.isna(tmdb_id):
        return None
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}",
            timeout=5
        )
        if r.status_code != 200:
            return None
        p = r.json().get("poster_path")
        if p:
            return f"https://image.tmdb.org/t/p/w400{p}"
    except:
        return None
    return None

# -------------------- LOAD NMF --------------------
@st.cache_data
def load_nmf(path):
    d = np.load(path, allow_pickle=True)
    return d["H"], d["movie_ids"].astype(int)

H, nmf_movie_ids = load_nmf(NMF_MODEL_PATH)
H_pinv = pinv(H)

# =========================================================
# RATE MOVIES
# =========================================================
st.subheader("Rate Movies")

movie_search = st.text_input("Search Movie Title")
filtered_titles = (
    movies[movies["title"].str.contains(movie_search, case=False, na=False)]["title"].tolist()
    if movie_search else []
)

selected_movie = st.selectbox("Select Movie", filtered_titles if filtered_titles else ["No results"])
rating_val = st.slider("Rating", 1, 5, 3)

if st.button("Add Rating") and selected_movie != "No results":
    m_id = title_to_id[selected_movie]
    st.session_state.user_ratings[m_id] = rating_val

# SHOW USER RATINGS
if st.session_state.user_ratings:
    st.write("### Your Ratings")
    for m_id, r in st.session_state.user_ratings.items():
        st.write(f"{movie_dict[m_id]['title']} — ⭐ {r}")

    if st.button("Clear Ratings"):
        st.session_state.user_ratings = {}
        st.session_state.nmf_rec_index = 0
        st.rerun()

st.divider()

# =========================================================
# NMF RECOMMENDATIONS
# =========================================================
st.subheader("NMF Recommendations")

if st.button("Get Recommendations") and st.session_state.user_ratings:

    st.session_state.nmf_rec_index = 0  # reset paging

    # BUILD USER VECTOR
    user_vec = pd.Series(0, index=nmf_movie_ids, dtype=float)
    for m_id, r in st.session_state.user_ratings.items():
        if m_id in user_vec.index:
            user_vec[m_id] = r

    mean_val = user_vec[user_vec > 0].mean() if (user_vec > 0).any() else 0
    centered = (user_vec - mean_val).fillna(0)

    latent = np.dot(centered.values, H_pinv)
    preds = np.dot(latent, H) + mean_val
    preds = np.clip(preds, 1, 5)

    preds_series = pd.Series(preds, index=nmf_movie_ids)
    preds_series = preds_series.drop(
        labels=[m for m in st.session_state.user_ratings if m in preds_series.index]
    )
    preds_series = preds_series.sort_values(ascending=False)

    st.session_state["preds_series"] = preds_series

# DISPLAY RESULTS IF AVAILABLE
if "preds_series" in st.session_state:

    preds_series = st.session_state["preds_series"]
    start = st.session_state.nmf_rec_index
    end = start + BATCH_SIZE

    for m_id, pred in preds_series.iloc[start:end].items():
        if m_id not in movie_dict:
            continue

        row = movie_dict[m_id]
        poster = get_poster(row["tmdbId"])

        if poster:
            st.image(poster, caption=f"{row['title']} ⭐ {pred:.2f}")
        else:
            st.write(f"**{row['title']}** — ⭐ {pred:.2f}")

    st.session_state.nmf_rec_index += BATCH_SIZE

    if st.session_state.nmf_rec_index < len(preds_series):
        if st.button("Load More"):
            pass
    else:
        st.session_state.nmf_rec_index = 0
