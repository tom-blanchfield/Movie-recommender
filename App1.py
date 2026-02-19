import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import pinv

st.set_page_config(page_title="Movie Recommender 2.0", layout="wide")
st.title("🎬 MoRiS 2.0 — Movie Recommender")

# -------------------- SETTINGS --------------------
TMDB_API_KEY = "888bb40cd1f4d3c95b375753e9c34c09"
RATINGS_PATH = "ratings.csv"
MOVIES_PATH = "movies.csv"
LINKS_PATH = "links.csv"
TAGS_PATH = "tags.csv"
NMF_MODEL_PATH = "nmf_300f_top3000.npz"

MIN_RATINGS = 20
MIN_REC_RATINGS = 10
MIN_OVERLAP = 5

# 🔥 New constants (only change requested)
MIN_PRED_RATING = 4.0
MAX_RECS = 150
BATCH_SIZE = 30

# -------------------- LOAD DATA --------------------
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

rating_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
rating_stats.columns = ["movieId", "avg_rating", "rating_count"]
movies = movies.merge(rating_stats, on="movieId", how="left")
movies["avg_rating"] = movies["avg_rating"].fillna(0)
movies["rating_count"] = movies["rating_count"].fillna(0)

user_movie_matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")
user_means = user_movie_matrix.mean(axis=1)
mean_centered = user_movie_matrix.sub(user_means, axis=0).fillna(0)

# -------------------- SESSION STATE --------------------
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

if "nmf_rec_index" not in st.session_state:
    st.session_state.nmf_rec_index = 0

if "collab_rec_index" not in st.session_state:
    st.session_state.collab_rec_index = 0

# -------------------- POSTER --------------------
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
    except:
        return None
    return None

# -------------------- LOAD NMF --------------------
@st.cache_data
def load_nmf_model(path):
    data = np.load(path, allow_pickle=True)
    return {
        "W": data["W"],
        "H": data["H"],
        "user_ids": data["user_ids"],
        "movie_ids": data["movie_ids"]
    }

nmf_data = load_nmf_model(NMF_MODEL_PATH)
H = nmf_data["H"]
nmf_movie_ids = nmf_data["movie_ids"]

# =========================================================
# RATE MOVIES
# =========================================================
st.subheader("Rate Movies")

movie_search = st.text_input("Type part of a movie title")

filtered_titles = movies[
    movies["title"].str.contains(movie_search, case=False, na=False)
]["title"].tolist() if movie_search else []

selected_movie = st.selectbox(
    "Select movie",
    options=filtered_titles if filtered_titles else ["No results"]
)

rating_value = st.slider("Rating", 1, 5, 3)

if st.button("Add Rating") and filtered_titles:
    movie_id = int(
        movies[movies["title"] == selected_movie]["movieId"].values[0]
    )
    st.session_state.user_ratings[movie_id] = int(rating_value)

if st.session_state.user_ratings:
    st.write("### Your Ratings")
    for m_id, r in st.session_state.user_ratings.items():
        title = movies[movies["movieId"] == m_id]["title"].values[0]
        st.write(f"{title}: {r}")

st.divider()

# =========================================================
# NMF RECOMMENDATIONS (FIXED AS REQUESTED)
# =========================================================

if st.button("Get NMF Recommendations") and st.session_state.user_ratings:

    st.session_state.nmf_rec_index = 0

    user_vec = pd.Series(0, index=nmf_movie_ids)

    for m_id, r in st.session_state.user_ratings.items():
        if m_id in user_vec.index:
            user_vec[m_id] = r

    user_mean = user_vec[user_vec > 0].mean() if (user_vec > 0).any() else 0
    user_vec_centered = (user_vec - user_mean).fillna(0)

    user_latent = np.dot(user_vec_centered.values, pinv(H))
    preds = np.dot(user_latent, H) + user_mean
    preds = np.clip(preds, 1, 5)

    preds_series = pd.Series(preds, index=nmf_movie_ids)

    preds_series = preds_series.drop(
        labels=[m for m in st.session_state.user_ratings.keys() if m in preds_series.index],
        errors="ignore"
    )

    # 🔥 Only predicted 4+
    preds_series = preds_series[preds_series >= MIN_PRED_RATING]

    preds_series = preds_series.sort_values(ascending=False)

    preds_series = preds_series.head(MAX_RECS)

    st.session_state.nmf_recs = preds_series

# -------------------- DISPLAY NMF RECS --------------------

if "nmf_recs" in st.session_state:

    recs = st.session_state.nmf_recs

    start = st.session_state.nmf_rec_index
    end = start + BATCH_SIZE

    batch = recs.iloc[start:end]

    if batch.empty:
        st.info("No more recommendations.")
    else:
        for m_id, pred_rating in zip(batch.index, batch.values):

            row = movies[movies["movieId"] == m_id].iloc[0]
            poster = get_poster(row["tmdbId"])

            if poster:
                st.markdown(
                    f"<div style='text-align:center'>"
                    f"<img src='{poster}' width='250'><br>"
                    f"<strong>{row['title']}</strong><br>"
                    f"⭐ Predicted: {pred_rating:.2f}"
                    f"</div>",
                    unsafe_allow_html=True
                )

        if end < len(recs):
            if st.button("Load 30 More"):
                st.session_state.nmf_rec_index += BATCH_SIZE
                st.rerun()

# =========================================================
# COLLABORATIVE FILTERING (UNCHANGED)
# =========================================================

st.divider()
st.subheader("Collaborative Filtering Recommendations")

if st.button("Get Collaborative Recommendations") and st.session_state.user_ratings:

    st.session_state.collab_rec_index = 0

    new_user = pd.Series(np.nan, index=user_movie_matrix.columns)

    for m_id, r in st.session_state.user_ratings.items():
        if m_id in new_user.index:
            new_user[m_id] = r

    temp_matrix = user_movie_matrix.copy()
    temp_matrix.loc["new_user"] = new_user

    temp_means = temp_matrix.mean(axis=1)
    temp_centered = temp_matrix.sub(temp_means, axis=0).fillna(0)

    similarity = cosine_similarity(
        temp_centered.loc[["new_user"]],
        temp_centered.drop("new_user")
    )[0]

    sim_series = pd.Series(similarity, index=temp_centered.drop("new_user").index)

    top_users = sim_series.sort_values(ascending=False).head(20).index

    weighted_sum = temp_centered.loc[top_users].T.dot(
        sim_series.loc[top_users]
    )

    sim_total = sim_series.loc[top_users].sum()

    collab_preds = weighted_sum / sim_total
    collab_preds = collab_preds + temp_means["new_user"]

    collab_preds = collab_preds.drop(
        labels=st.session_state.user_ratings.keys(),
        errors="ignore"
    )

    collab_preds = collab_preds.sort_values(ascending=False)

    st.session_state.collab_recs = collab_preds.head(150)

if "collab_recs" in st.session_state:

    recs = st.session_state.collab_recs

    start = st.session_state.collab_rec_index
    end = start + 30

    batch = recs.iloc[start:end]

    for m_id, pred_rating in zip(batch.index, batch.values):

        row = movies[movies["movieId"] == m_id].iloc[0]
        poster = get_poster(row["tmdbId"])

        if poster:
            st.markdown(
                f"<div style='text-align:center'>"
                f"<img src='{poster}' width='250'><br>"
                f"<strong>{row['title']}</strong><br>"
                f"⭐ Predicted: {pred_rating:.2f}"
                f"</div>",
                unsafe_allow_html=True
            )

    if end < len(recs):
        if st.button("Load 30 More (Collaborative)"):
            st.session_state.collab_rec_index += 30
            st.rerun()
