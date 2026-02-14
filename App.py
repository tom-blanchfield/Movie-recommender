import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender")

# ---------- AUTO SCREEN WIDTH ----------
st.markdown(
    """
    <script>
    const width = window.innerWidth;
    window.parent.postMessage({type: "streamlit:setSessionState", key: "screen_width", value: width}, "*");
    </script>
    """,
    unsafe_allow_html=True
)

if "screen_width" not in st.session_state:
    st.session_state.screen_width = 800

columns_num = 1 if st.session_state.screen_width < 900 else 3
poster_width = 350 if columns_num == 1 else 180

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

# ---------- RATINGS STATS ----------
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

    genre_tag_movies["genre_score"] = genre_tag_movies["genres"].apply(
        lambda g: sum(1 for sel in selected_genres if sel in g)
    )

    ranked = genre_tag_movies[
        (genre_tag_movies["genre_score"]>0) &
        (genre_tag_movies["rating_count"]>=MIN_RATINGS)
    ].sort_values(by=["avg_rating","rating_count"], ascending=False)

else:
    selected_tags = st.multiselect(
        "Enter Keywords", options=[], default=[], accept_new_options=True
    )
    selected_tags = [t.lower() for t in selected_tags]

    if selected_tags:
        mask = tags["tag"].str.lower().apply(
            lambda t: any(sel in t for sel in selected_tags)
        )
        tag_counts = tags[mask]["movieId"].value_counts()
        genre_tag_movies["tag_score"] = genre_tag_movies["movieId"].map(tag_counts).fillna(0)
    else:
        genre_tag_movies["tag_score"] = 0

    ranked = genre_tag_movies[
        (genre_tag_movies["tag_score"]>0) &
        (genre_tag_movies["rating_count"]>=MIN_RATINGS)
    ].sort_values(by=["avg_rating","rating_count"], ascending=False)

if not ranked.empty:
    for i in range(0, min(30,len(ranked)), columns_num):
        cols = st.columns(columns_num)
        for col, (_, row) in zip(cols, ranked.iloc[i:i+columns_num].iterrows()):
            with col:
                poster = get_poster(row["tmdbId"])
                if poster:
                    st.image(poster, width=poster_width)
                st.markdown(f"**{row['title']}**  \n⭐ {row['avg_rating']:.2f}")

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

    for i in range(0, len(recs), columns_num):
        cols = st.columns(columns_num)
        for col, (_, row) in zip(cols, recs.iloc[i:i+columns_num].iterrows()):
            with col:
                poster = get_poster(row["tmdbId"])
                if poster:
                    st.image(poster, width=poster_width)
                st.markdown(f"**{row['title']}**")
