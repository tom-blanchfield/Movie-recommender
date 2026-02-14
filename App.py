import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender")

# ---------- CSS GRID RESPONSIVE ----------
st.markdown("""
<style>
.poster-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 20px;
    justify-items: center;
}
@media (min-width: 900px) {
    .poster-grid {
        grid-template-columns: repeat(3, 1fr);
    }
}
.poster img {
    width: 220px;
    border-radius: 10px;
}
@media (max-width: 899px) {
    .poster img {
        width: 360px;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------- TMDB API ----------------
TMDB_API_KEY = "YOUR_API_KEY"

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
rating_stats = ratings.groupby("movieId")["rating"].agg(["mean","count"]).reset_index()
rating_stats.columns = ["movieId","avg_rating","rating_count"]
movies = movies.merge(rating_stats, on="movieId", how="left")
movies.fillna({"avg_rating":0,"rating_count":0}, inplace=True)

MIN_RATINGS = 20

# ---------- POSTER ----------
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
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        return None
    return None

# ---------- GENRE LIST ----------
all_genres = sorted(
    set(g for sub in movies["genres"].str.split("|") for g in sub if g and g!="(no genres listed)")
)

# ---------- TOP 50% USERS ----------
user_counts = ratings["userId"].value_counts()
top_users = user_counts.head(int(len(user_counts)*0.5)).index
ratings_top = ratings[ratings["userId"].isin(top_users)]

user_movie_matrix = ratings_top.pivot_table(
    index="userId", columns="movieId", values="rating"
).fillna(0)

# ---------- SESSION ----------
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

# =================================================
# DISCOVERY
# =================================================
st.subheader("Discover Movies")

mode = st.selectbox("Recommendation Mode", ["Genres","Keywords"])
genre_tag_movies = movies.copy()

if mode == "Genres":
    selected_genres = st.multiselect("Select Genres", all_genres)
    genre_tag_movies["score"] = genre_tag_movies["genres"].apply(
        lambda g: sum(1 for sel in selected_genres if sel in g)
    )
else:
    selected_tags = st.multiselect("Enter Keywords", options=[], accept_new_options=True)
    selected_tags = [t.lower() for t in selected_tags]

    if selected_tags:
        mask = tags["tag"].str.lower().apply(
            lambda t: any(sel in t for sel in selected_tags)
        )
        tag_counts = tags[mask]["movieId"].value_counts()
        genre_tag_movies["score"] = genre_tag_movies["movieId"].map(tag_counts).fillna(0)
    else:
        genre_tag_movies["score"] = 0

ranked = genre_tag_movies[
    (genre_tag_movies["score"]>0) &
    (genre_tag_movies["rating_count"]>=MIN_RATINGS)
].sort_values(by=["avg_rating","rating_count"], ascending=False)

if not ranked.empty:
    st.markdown('<div class="poster-grid">', unsafe_allow_html=True)
    for _, row in ranked.head(30).iterrows():
        poster = get_poster(row["tmdbId"])
        if poster:
            st.markdown(
                f'<div class="poster"><img src="{poster}"><br><b>{row["title"]}</b></div>',
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# =================================================
# RATE MOVIES
# =================================================
st.subheader("Rate Movies")

search = st.text_input("Search movie title")
matches = movies[movies["title"].str.contains(search, case=False, na=False)].head(15)

selected_movie = st.selectbox("Select movie", matches["title"] if not matches.empty else ["No results"])
rating_value = st.slider("Rating",1,5,3)

if st.button("Add Rating") and not matches.empty:
    m_id = int(movies[movies["title"]==selected_movie]["movieId"].values[0])
    st.session_state.user_ratings[m_id] = rating_value

# =================================================
# COLLAB RECOMMENDER
# =================================================
if st.button("Get Recommendations") and st.session_state.user_ratings:

    user_vector = np.zeros(user_movie_matrix.shape[1])
    movie_map = {int(m):i for i,m in enumerate(user_movie_matrix.columns)}

    for m_id, r in st.session_state.user_ratings.items():
        if m_id in movie_map:
            user_vector[movie_map[m_id]] = r

    similarities = cosine_similarity([user_vector], user_movie_matrix.values)[0]
    similar_users = user_movie_matrix.index[np.argsort(similarities)[-10:]]

    fav_movies = ratings_top[
        (ratings_top["userId"].isin(similar_users)) &
        (ratings_top["rating"]>=4)
    ]

    movie_scores = fav_movies["movieId"].value_counts()
    movie_scores = movie_scores[
        ~movie_scores.index.isin(st.session_state.user_ratings.keys())
    ].head(30)

    recs = movies.set_index("movieId").loc[movie_scores.index]

    st.subheader("Recommended Movies")
    st.markdown('<div class="poster-grid">', unsafe_allow_html=True)

    for m_id in recs.index:
        row = recs.loc[m_id]
        poster = get_poster(row["tmdbId"])
        if poster:
            st.markdown(
                f'<div class="poster"><img src="{poster}"><br><b>{row["title"]}</b></div>',
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)
