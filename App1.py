import streamlit as st
import pandas as pd
import numpy as np
import requests
from numpy.linalg import pinv

st.set_page_config(page_title="Movie Recommender 2.0", layout="wide")
st.title("🎬 MoRiS 2.0 — Movie Recommender")

# ---------------- SETTINGS ----------------
TMDB_API_KEY = "YOUR_KEY"
NMF_MODEL_PATH = "nmf_top3000_300f.npz"
BATCH_SIZE = 30
MAX_POOL = 150

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    links = pd.read_csv("links.csv")
    return ratings, movies, links

ratings, movies, links = load_data()

movies = movies.merge(links, on="movieId", how="left")
movies["genres"] = movies["genres"].fillna("")

rating_stats = ratings.groupby("movieId")["rating"].agg(["mean","count"]).reset_index()
rating_stats.columns = ["movieId","avg_rating","rating_count"]
movies = movies.merge(rating_stats, on="movieId", how="left")
movies.fillna({"avg_rating":0,"rating_count":0}, inplace=True)

# ---------------- FAST TITLE SEARCH STRUCTURES ----------------
@st.cache_data
def build_title_structures(df):
    titles = df["title"].tolist()
    titles_lower = [t.lower() for t in titles]
    title_to_id = dict(zip(df["title"], df["movieId"]))
    id_to_title = dict(zip(df["movieId"], df["title"]))
    return titles, titles_lower, title_to_id, id_to_title

titles, titles_lower, title_to_id, id_to_title = build_title_structures(movies)

# ---------------- SESSION ----------------
defaults = {
    "user_ratings": {},
    "nmf_pool": [],
    "nmf_index": 0
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- POSTERS ----------------
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

# ---------------- LOAD NMF ----------------
@st.cache_data
def load_nmf(path):
    d = np.load(path, allow_pickle=True)
    return d["H"], d["movie_ids"].astype(int)

H, nmf_movie_ids = load_nmf(NMF_MODEL_PATH)
H_pinv = pinv(H)

# Only movies that exist in both model and movies.csv
valid_model_movie_ids = set(nmf_movie_ids).intersection(set(movies["movieId"]))

st.divider()

# =========================================================
# RATE MOVIES
# =========================================================
st.subheader("Rate Movies")

search = st.text_input("Type part of a movie title").lower()

if search:
    matches = [
        titles[i]
        for i, t in enumerate(titles_lower)
        if search in t
    ][:50]
else:
    matches = []

selected_movie = st.selectbox(
    "Select movie",
    options=matches if matches else ["No results"]
)

rating_val = st.slider("Rating",1,5,3)

if st.button("Add Rating") and matches:
    m_id = int(title_to_id[selected_movie])
    st.session_state.user_ratings[m_id] = rating_val

# SHOW RATINGS
if st.session_state.user_ratings:
    st.write("### Your Ratings")
    for m_id, r in st.session_state.user_ratings.items():
        st.write(f"{id_to_title[m_id]} ⭐ {r}")

st.divider()

# =========================================================
# NMF RECOMMENDATIONS
# =========================================================
if st.button("Get NMF Recommendations") and st.session_state.user_ratings:

    st.session_state.nmf_index = 0

    # Only include movies model knows
    user_vec = pd.Series(0, index=nmf_movie_ids, dtype=float)
    for m_id, r in st.session_state.user_ratings.items():
        if m_id in user_vec.index:
            user_vec[m_id] = r

    if user_vec.sum() == 0:
        st.warning("None of your rated movies exist in the model.")
    else:
        mean_val = user_vec[user_vec>0].mean()
        centered = (user_vec-mean_val).fillna(0)

        latent = np.dot(centered.values, H_pinv)
        preds = np.dot(latent, H) + mean_val
        preds = np.clip(preds,1,5)

        preds_series = pd.Series(preds,index=nmf_movie_ids)

        # Remove rated movies
        preds_series = preds_series.drop(
            labels=[m for m in st.session_state.user_ratings if m in preds_series.index]
        )

        # Only valid movies
        preds_series = preds_series[preds_series.index.isin(valid_model_movie_ids)]

        preds_series = preds_series.sort_values(ascending=False)

        st.session_state.nmf_pool = list(preds_series.head(MAX_POOL).items())

# DISPLAY
st.subheader("NMF Recommendations")

pool = st.session_state.nmf_pool
end = st.session_state.nmf_index + BATCH_SIZE

for m_id, pred in pool[st.session_state.nmf_index:end]:
    row = movies[movies["movieId"]==m_id]
    if row.empty:
        continue
    row = row.iloc[0]
    poster = get_poster(row["tmdbId"])
    if poster:
        st.markdown(
            f"<div style='text-align:center'>"
            f"<img src='{poster}' width='300'><br>"
            f"<strong>{row['title']}</strong><br>"
            f"⭐ {pred:.2f}</div>",
            unsafe_allow_html=True
        )

if end < len(pool):
    if st.button("Load More"):
        st.session_state.nmf_index += BATCH_SIZE
        st.rerun()
