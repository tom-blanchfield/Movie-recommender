import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import pinv

st.set_page_config(page_title="Movie Recommender 2.0", layout="wide")
st.title("🎬 MoRiS 2.0 — Movie Recommender")

# -------------------- SETTINGS --------------------
TMDB_API_KEY = "YOUR_KEY"
RATINGS_PATH = "ratings.csv"
MOVIES_PATH = "movies.csv"
LINKS_PATH = "links.csv"
TAGS_PATH = "tags.csv"
NMF_MODEL_PATH = "nmf_300f_top3000.npz"

MIN_RATINGS = 20
MIN_REC_RATINGS = 10
MIN_OVERLAP = 5
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
movies.fillna({"avg_rating": 0, "rating_count": 0}, inplace=True)

user_movie_matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")
user_means = user_movie_matrix.mean(axis=1)
mean_centered = user_movie_matrix.sub(user_means, axis=0).fillna(0)

all_genres = sorted(set(g for sub in movies["genres"].str.split("|") for g in sub if g and g != "(no genres listed)"))

# -------------------- SESSION --------------------
for key in ["user_ratings","nmf_rec_index","collab_rec_index","genre_rec_index"]:
    if key not in st.session_state:
        st.session_state[key] = {} if key=="user_ratings" else 0

# -------------------- POSTER --------------------
@st.cache_data(show_spinner=False)
def get_poster(tmdb_id):
    if pd.isna(tmdb_id): return None
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}", timeout=5)
        if r.status_code != 200: return None
        p = r.json().get("poster_path")
        if p: return f"https://image.tmdb.org/t/p/w400{p}"
    except: return None
    return None

# -------------------- NMF LOAD --------------------
@st.cache_data
def load_nmf(path):
    data = np.load(path, allow_pickle=True)
    return data["H"], data["movie_ids"].astype(int)

H, nmf_movie_ids = load_nmf(NMF_MODEL_PATH)
H_pinv = pinv(H)

# =========================================================
# RATE MOVIES
# =========================================================
st.subheader("Rate Movies")
movie_search = st.text_input("Search")
filtered = movies[movies["title"].str.contains(movie_search, case=False, na=False)]["title"].tolist() if movie_search else []
selected = st.selectbox("Movie", filtered if filtered else ["No results"])
rating_val = st.slider("Rating",1,5,3)

if st.button("Add Rating") and filtered:
    m_id = int(movies[movies["title"]==selected]["movieId"].values[0])
    st.session_state.user_ratings[m_id] = rating_val

# =========================================================
# NMF RECOMMENDATIONS
# =========================================================
if st.button("Get NMF Recommendations") and st.session_state.user_ratings:

    user_vec = pd.Series(0, index=nmf_movie_ids, dtype=float)
    for m_id,r in st.session_state.user_ratings.items():
        if m_id in user_vec.index:
            user_vec[m_id] = r

    mean_val = user_vec[user_vec>0].mean() if (user_vec>0).any() else 0
    centered = (user_vec-mean_val).fillna(0)

    latent = np.dot(centered.values, H_pinv)
    preds = np.dot(latent, H)+mean_val
    preds = np.clip(preds,1,5)

    preds_series = pd.Series(preds,index=nmf_movie_ids)
    preds_series = preds_series.drop(labels=[m for m in st.session_state.user_ratings if m in preds_series.index])
    preds_series = preds_series.sort_values(ascending=False)

    start = st.session_state.nmf_rec_index
    end = start + BATCH_SIZE

    for m_id,pred in preds_series.iloc[start:end].items():
        row = movies[movies["movieId"]==m_id]
        if row.empty: continue
        row=row.iloc[0]
        poster=get_poster(row["tmdbId"])
        if poster:
            st.image(poster,caption=f"{row['title']} ⭐ {pred:.2f}")

    st.session_state.nmf_rec_index += BATCH_SIZE
