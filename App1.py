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
BATCH_SIZE = 30  # for pagination of recommendations

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

# Rating stats
rating_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
rating_stats.columns = ["movieId", "avg_rating", "rating_count"]
movies = movies.merge(rating_stats, on="movieId", how="left")
movies["avg_rating"] = movies["avg_rating"].fillna(0)
movies["rating_count"] = movies["rating_count"].fillna(0)

# User matrices
user_movie_matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")
user_means = user_movie_matrix.mean(axis=1)
mean_centered = user_movie_matrix.sub(user_means, axis=0).fillna(0)

# Genre list
all_genres = sorted(set(g for sub in movies["genres"].str.split("|") for g in sub if g and g != "(no genres listed)"))

# Session state
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}
if "nmf_rec_index" not in st.session_state:
    st.session_state.nmf_rec_index = 0
if "collab_rec_index" not in st.session_state:
    st.session_state.collab_rec_index = 0
if "genre_rec_index" not in st.session_state:
    st.session_state.genre_rec_index = 0

# -------------------- POSTER FUNCTION --------------------
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

# -------------------- NMF MODEL --------------------
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
W = nmf_data["W"]
H = nmf_data["H"]
nmf_user_ids = nmf_data["user_ids"]
nmf_movie_ids = nmf_data["movie_ids"]

# =========================================================
# DISCOVER MOVIES
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

genre_tag_movies["total_score"] = genre_tag_movies["genre_score"] + genre_tag_movies["tag_score"]

if (mode == "Genres" and selected_genres) or (mode == "Keywords" and selected_tags):
    ranked_movies = genre_tag_movies[
        (genre_tag_movies["total_score"] > 0) &
        (genre_tag_movies["rating_count"] >= MIN_REC_RATINGS)
    ].sort_values(by=["avg_rating", "rating_count"], ascending=False)

    start = st.session_state.genre_rec_index
    end = start + BATCH_SIZE
    for _, row in ranked_movies.iloc[start:end].iterrows():
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
    st.session_state.genre_rec_index += BATCH_SIZE
    if st.session_state.genre_rec_index < len(ranked_movies):
        if st.button("Load More Genre/Keyword Recommendations"):
            pass
    else:
        st.session_state.genre_rec_index = 0

st.divider()

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

st.divider()

# =========================================================
# COLLABORATIVE FILTERING RECOMMENDATIONS
# =========================================================
if st.button("Get Collaborative Recommendations") and st.session_state.user_ratings:
    user_vector = pd.Series(0, index=mean_centered.columns, dtype=float)
    for m_id, r in st.session_state.user_ratings.items():
        if m_id in user_vector.index:
            user_vector[m_id] = r

    target_mean = user_vector[user_vector > 0].mean() if (user_vector > 0).any() else 0
    user_vector_centered = (user_vector - target_mean).fillna(0)

    similarities = cosine_similarity([user_vector_centered], mean_centered.values)[0]
    overlaps = (mean_centered != 0).dot((user_vector_centered != 0).astype(int))
    valid_users = np.where(overlaps >= MIN_OVERLAP)[0]

    similarities_filtered = similarities[valid_users]
    top_idx = valid_users[np.argsort(similarities_filtered)[-10:]]

    preds = {}
    for movie_id in mean_centered.columns:
        if movie_id in st.session_state.user_ratings:
            continue
        movie_info = movies[movies["movieId"] == movie_id]
        if movie_info.empty or movie_info.iloc[0]["rating_count"] < MIN_REC_RATINGS:
            continue
        num = 0
        den = 0
        for idx in top_idx:
            sim = similarities[idx]
            if sim <= 0:
                continue
            rating = mean_centered.iloc[idx][movie_id]
            if rating != 0:
                num += sim * rating
                den += abs(sim)
        if den > 0:
            preds[movie_id] = target_mean + (num / den)

    preds_series = pd.Series(preds).sort_values(ascending=False)

    start = st.session_state.collab_rec_index
    end = start + BATCH_SIZE
    for m_id in preds_series.index[start:end]:
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
    st.session_state.collab_rec_index += BATCH_SIZE
    if st.session_state.collab_rec_index < len(preds_series):
        if st.button("Load More Collaborative Recommendations"):
            pass
    else:
        st.session_state.collab_rec_index = 0

st.divider()

# =========================================================
# NMF RECOMMENDATIONS
# =========================================================
if st.button("Get NMF Recommendations") and st.session_state.user_ratings:

    # Build user vector
    user_vec = pd.Series(0, index=nmf_movie_ids, dtype=float)
    for m_id, r in st.session_state.user_ratings.items():
        if m_id in user_vec.index:
            user_vec[m_id] = r

    # 🔥 Project into latent space properly
    user_latent = np.dot(user_vec.values, H.T)

    # 🔥 Reconstruct ratings
    preds = np.dot(user_latent, H)

    # Scale back to 1–5 range
    preds = np.clip(preds, 1, 5)

    preds_series = pd.Series(preds, index=nmf_movie_ids)

    # Remove already rated movies
    preds_series = preds_series.drop(
        labels=[m for m in st.session_state.user_ratings.keys() if m in preds_series.index],
        errors="ignore"
    )

    # Sort highest first
    preds_series = preds_series.sort_values(ascending=False)

    if preds_series.empty:
        st.warning("No recommendations available.")
        st.stop()

    start = st.session_state.nmf_rec_index
    end = start + BATCH_SIZE

    for m_id, pred_rating in zip(preds_series.index[start:end], preds_series.values[start:end]):
        row = movies[movies["movieId"] == m_id].iloc[0]
        poster = get_poster(row["tmdbId"])
        if poster:
            st.markdown(
                f"<div style='text-align:center'>"
                f"<img src='{poster}' width='300'><br>"
                f"<strong>{row['title']}</strong><br>"
                f"⭐ {pred_rating:.2f}"
                f"</div>",
                unsafe_allow_html=True
            )

    st.session_state.nmf_rec_index += BATCH_SIZE
    if st.session_state.nmf_rec_index < len(preds_series):
        if st.button("Load More NMF Recommendations"):
            pass
    else:
        st.session_state.nmf_rec_index = 0
