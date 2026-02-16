import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# CONFIG — QUICK TUNING VARIABLES
# =========================================================
MIN_RATINGS = 20
MIN_REC_RATINGS = 10
MIN_OVERLAP = 5
NUM_NEIGHBORS = 10

DISCOVERY_BATCH_SIZE = 30
REC_BATCH_SIZE = 30
MAX_POOL_SIZE = 150

TMDB_API_KEY = "888bb40cd1f4d3c95b375753e9c34c09"

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender")

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

rating_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
rating_stats.columns = ["movieId", "avg_rating", "rating_count"]

movies = movies.merge(rating_stats, on="movieId", how="left")
movies["avg_rating"] = movies["avg_rating"].fillna(0)
movies["rating_count"] = movies["rating_count"].fillna(0)

# ---------- POSTERS ----------
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

# ---------- GENRES ----------
all_genres = sorted(set(g for sub in movies["genres"].str.split("|") for g in sub if g and g != "(no genres listed)"))

# ---------- MATRICES ----------
user_movie_matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")
user_means = user_movie_matrix.mean(axis=1)
mean_centered = user_movie_matrix.sub(user_means, axis=0).fillna(0)

# ---------- SESSION STATE ----------
defaults = {
    "user_ratings": {},
    "disc_index": 0,
    "rec_index": 0,
    "disc_pool": [],
    "rec_pool": []
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# DISCOVER MOVIES
# =========================================================
st.subheader("Discover Movies")

mode = st.selectbox("Choose recommendation mode", ["Genres", "Keywords"])
genre_tag_movies = movies.copy()

selected_genres = []
selected_tags = []

if mode == "Genres":
    selected_genres = st.multiselect("Select Genres", all_genres)
    genre_tag_movies["genre_score"] = genre_tag_movies["genres"].apply(
        lambda g: sum(1 for sel in selected_genres if sel in g)
    )
    genre_tag_movies["tag_score"] = 0

else:
    selected_tags = st.multiselect("Keywords", options=[], default=[], accept_new_options=True)
    selected_tags = [t.lower() for t in selected_tags]
    genre_tag_movies["genre_score"] = 0
    if selected_tags:
        tag_mask = tags["tag"].str.lower().apply(lambda t: any(sel in t for sel in selected_tags))
        tag_counts = tags[tag_mask]["movieId"].value_counts()
        genre_tag_movies["tag_score"] = genre_tag_movies["movieId"].map(tag_counts).fillna(0)
    else:
        genre_tag_movies["tag_score"] = 0

genre_tag_movies["total_score"] = genre_tag_movies["genre_score"] + genre_tag_movies["tag_score"]

if (selected_genres or selected_tags):

    ranked_movies = genre_tag_movies[
        (genre_tag_movies["total_score"] > 0) &
        (genre_tag_movies["rating_count"] >= MIN_REC_RATINGS)
    ].sort_values(by=["avg_rating","rating_count"], ascending=False).head(MAX_POOL_SIZE)

    st.session_state.disc_pool = ranked_movies.to_dict("records")

end = st.session_state.disc_index + DISCOVERY_BATCH_SIZE
for row in st.session_state.disc_pool[st.session_state.disc_index:end]:
    poster = get_poster(row["tmdbId"])
    if poster:
        st.markdown(f"<div style='text-align:center'><img src='{poster}' width='300'><br><strong>{row['title']}</strong></div>", unsafe_allow_html=True)

if end < len(st.session_state.disc_pool):
    if st.button("Load More Discovery"):
        st.session_state.disc_index += DISCOVERY_BATCH_SIZE
        st.rerun()

st.divider()

# =========================================================
# RATE MOVIES
# =========================================================
st.subheader("Rate Movies")
movie_search = st.text_input("Type part of a movie title")

filtered_titles = movies[movies["title"].str.contains(movie_search, case=False, na=False)]["title"].tolist() if movie_search else []
selected_movie = st.selectbox("Select movie", options=filtered_titles if filtered_titles else ["No results"])
rating_value = st.slider("Rating",1,5,3)

if st.button("Add Rating") and filtered_titles:
    movie_id = int(movies[movies["title"]==selected_movie]["movieId"].values[0])
    st.session_state.user_ratings[movie_id] = rating_value

# =========================================================
# RECOMMENDATIONS
# =========================================================
if st.button("Get Recommendations") and st.session_state.user_ratings:

    st.session_state.rec_index = 0

    user_vector = pd.Series(0,index=mean_centered.columns)
    for m,r in st.session_state.user_ratings.items():
        if m in user_vector.index:
            user_vector[m]=r

    target_mean = user_vector[user_vector>0].mean()
    user_vector = (user_vector-target_mean).fillna(0)

    sims = cosine_similarity([user_vector],mean_centered.values)[0]
    overlaps = (mean_centered!=0).dot((user_vector!=0).astype(int))
    valid = np.where(overlaps>=MIN_OVERLAP)[0]

    sims_filtered = sims[valid]
    top_idx = valid[np.argsort(sims_filtered)[-NUM_NEIGHBORS:]]

    preds={}
    for movie_id in mean_centered.columns:
        if movie_id in st.session_state.user_ratings: continue
        movie_info = movies[movies["movieId"]==movie_id]
        if movie_info.empty or movie_info.iloc[0]["rating_count"]<MIN_REC_RATINGS: continue

        num=den=0
        for idx in top_idx:
            sim=sims[idx]
            if sim<=0: continue
            rating=mean_centered.iloc[idx][movie_id]
            if rating!=0:
                num+=sim*rating
                den+=abs(sim)

        if den>0:
            preds[movie_id]=target_mean+(num/den)

    sorted_preds = sorted(preds.items(), key=lambda x:x[1], reverse=True)[:MAX_POOL_SIZE]
    st.session_state.rec_pool = sorted_preds

# ---------- DISPLAY REC BATCH ----------
st.subheader("Recommended Movies")

end = st.session_state.rec_index + REC_BATCH_SIZE
for m_id,_ in st.session_state.rec_pool[st.session_state.rec_index:end]:
    row = movies[movies["movieId"]==m_id].iloc[0]
    poster = get_poster(row["tmdbId"])
    if poster:
        st.markdown(f"<div style='text-align:center'><img src='{poster}' width='300'><br><strong>{row['title']}</strong></div>", unsafe_allow_html=True)

if end < len(st.session_state.rec_pool):
    if st.button("Load More Recommendations"):
        st.session_state.rec_index += REC_BATCH_SIZE
        st.rerun()
