import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image

from sklearn.metrics.pairwise import cosine_similarity

# -------------------- SETTINGS --------------------
RATINGS_PATH = "/content/drive/MyDrive/ratings.csv"
MOVIES_PATH = "/content/drive/MyDrive/movies.csv"
LINKS_PATH = "/content/drive/MyDrive/links.csv"
NMF_MODEL_PATH = "/content/drive/MyDrive/nmf_top300x1000_50f.npz"

MIN_REC_RATINGS = 10
MIN_OVERLAP = 5
TOP_N_RECS = 30

# TMDB API
TMDB_API_KEY = "888bb40cd1f4d3c95b375753e9c34c09"

st.set_page_config(page_title="MoRiS 2.0 — NMF Recommender", layout="wide")
st.title("🎬 MoRiS 2.0 — NMF-Powered Movie Recommendations")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)
    links = pd.read_csv(LINKS_PATH)
    movies = movies.merge(links, on="movieId", how="left")
    return ratings, movies

ratings, movies = load_data()
movies["avg_rating"] = ratings.groupby("movieId")["rating"].mean().reindex(movies["movieId"]).fillna(0)
movies["rating_count"] = ratings.groupby("movieId")["rating"].count().reindex(movies["movieId"]).fillna(0)

# -------------------- LOAD NMF MODEL --------------------
@st.cache_data
def load_nmf_model(path=NMF_MODEL_PATH):
    data = np.load(path, allow_pickle=True)
    return data['W'], data['H'], data['user_ids'], data['movie_ids']

W, H, user_ids, movie_ids = load_nmf_model()
latent_factors = H.shape[0]

# -------------------- POSTER FETCH FUNCTION --------------------
@st.cache_data(show_spinner=False)
def get_poster(tmdb_id):
    if pd.isna(tmdb_id):
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}"
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        poster_path = r.json().get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w400{poster_path}"
    except Exception:
        return None
    return None

# -------------------- SESSION STATE --------------------
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

# -------------------- RATE MOVIES --------------------
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

# -------------------- NMF PREDICTIONS --------------------
def predict_nmf_ratings(user_ratings_dict):
    user_vector = np.zeros(len(movie_ids))
    for m_id, r in user_ratings_dict.items():
        if m_id in movie_ids:
            idx = np.where(movie_ids == m_id)[0][0]
            user_vector[idx] = r

    user_vector_centered = user_vector.copy()
    # Simple approach: project onto latent space via non-negative least squares
    from scipy.optimize import nnls
    user_features, _ = nnls(H.T, user_vector)
    predicted_ratings = user_features @ H
    return predicted_ratings

# -------------------- GET RECOMMENDATIONS --------------------
if st.button("Get Recommendations") and len(st.session_state.user_ratings) > 0:
    st.subheader("Recommended Movies")
    preds = predict_nmf_ratings(st.session_state.user_ratings)

    recs_df = pd.DataFrame({
        "movieId": movie_ids,
        "pred_rating": preds
    })
    # Filter out movies already rated by user
    recs_df = recs_df[~recs_df["movieId"].isin(st.session_state.user_ratings.keys())]
    # Filter by minimum ratings
    recs_df = recs_df.merge(movies[["movieId", "rating_count", "title", "tmdbId"]], on="movieId")
    recs_df = recs_df[recs_df["rating_count"] >= MIN_REC_RATINGS]
    recs_df = recs_df.sort_values("pred_rating", ascending=False).head(TOP_N_RECS)

    for _, row in recs_df.iterrows():
        poster = get_poster(row["tmdbId"])
        if poster:
            st.markdown(
                f"<div style='text-align:center'>"
                f"<img src='{poster}' width='300'><br>"
                f"<strong>{row['title']}</strong><br>"
                f"Predicted rating: {row['pred_rating']:.2f}"
                f"</div>",
                unsafe_allow_html=True
            )
