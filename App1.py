import streamlit as st
import pandas as pd
import numpy as np
import requests
from pathlib import Path

st.set_page_config(page_title="Movie Recommender 2.0 (NMF)", layout="wide")
st.title("🎬 MoRiS 2.0 — Movie Recommender (NMF)")

# ---------- SETTINGS ----------
TMDB_API_KEY = "888bb40cd1f4d3c95b375753e9c34c09"
MIN_REC_RATINGS = 10
NMF_MODEL_PATH = "nmf_top300x1000_50f.npz"  # <-- updated filename

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    links = pd.read_csv("links.csv")
    return ratings, movies, links

ratings, movies, links = load_data()
movies = movies.merge(links, on="movieId", how="left")
movies["genres"] = movies["genres"].fillna("")

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

# ---------- SESSION STATE ----------
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

# =========================================================
# LOAD NMF MODEL SAFELY
# =========================================================
nmf_pred_matrix = None
if Path(NMF_MODEL_PATH).exists():
    data = np.load(NMF_MODEL_PATH, allow_pickle=True)
    W = data["W"]
    H = data["H"]
    user_ids = data["user_ids"]
    movie_ids = data["movie_ids"]
    nmf_pred_matrix = pd.DataFrame(np.dot(W, H), index=user_ids, columns=movie_ids)
else:
    st.warning(f"NMF model not found at '{NMF_MODEL_PATH}'. Upload the file and refresh the app.")

# =========================================================
# RATE MOVIES
# =========================================================
st.subheader("Rate Movies")
movie_search = st.text_input("Type part of a movie title")
filtered_titles = movies[movies["title"].str.contains(movie_search, case=False, na=False)]["title"].tolist() if movie_search else []
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
# GET NMF RECOMMENDATIONS
# =========================================================
if st.button("Get NMF Recommendations"):
    if nmf_pred_matrix is None:
        st.error("Cannot generate recommendations — NMF model not loaded.")
    elif len(st.session_state.user_ratings) == 0:
        st.info("Please rate some movies first.")
    else:
        user_vec = pd.Series(0, index=nmf_pred_matrix.columns)
        for m_id, r in st.session_state.user_ratings.items():
            if m_id in user_vec.index:
                user_vec[m_id] = r
        user_mean = user_vec[user_vec > 0].mean() if (user_vec > 0).any() else 0
        user_vec_centered = (user_vec - user_mean).fillna(0)
        preds = np.dot(user_vec_centered.values, nmf_pred_matrix.values.T).flatten()
        preds_series = pd.Series(preds, index=nmf_pred_matrix.columns)
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
