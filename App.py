import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender")

# -------------------- TMDB API KEY --------------------
TMDB_API_KEY = "888bb40cd1f4d3c95b375753e9c34c09"

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
rating_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
rating_stats.columns = ["movieId", "avg_rating", "rating_count"]

movies = movies.merge(rating_stats, on="movieId", how="left")
movies["avg_rating"] = movies["avg_rating"].fillna(0)
movies["rating_count"] = movies["rating_count"].fillna(0)

MIN_RATINGS = 20
MIN_OVERLAP = 5   # <-- new

# ---------- POSTER FETCH FUNCTION ----------
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

# ---------- GENRE LIST ----------
all_genres = sorted(
    set(
        g
        for sub in movies["genres"].str.split("|")
        for g in sub
        if g and g != "(no genres listed)"
    )
)

# ---------- USER MATRICES ----------
user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

user_means = user_movie_matrix.mean(axis=1)
mean_centered = user_movie_matrix.sub(user_means, axis=0).fillna(0)

# ---------- SESSION STATE ----------
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

# =========================================================
# DISCOVER MOVIES (UNCHANGED)
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
        (genre_tag_movies["rating_count"] >= MIN_RATINGS)
    ].sort_values(by=["avg_rating", "rating_count"], ascending=False)

    for _, row in ranked_movies.head(30).iterrows():
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

st.divider()

# =========================================================
# RATE MOVIES (UNCHANGED)
# =========================================================
st.subheader("Rate Movies")

movie_search = st.text_input("Type part of a movie title")

if movie_search:
    filtered_titles = movies[movies["title"].str.contains(movie_search, case=False, na=False)]["title"].tolist()
else:
    filtered_titles = []

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
# IMPROVED COLLAB RECOMMENDATIONS
# =========================================================
if st.button("Get Recommendations") and len(st.session_state.user_ratings) > 0:

    # Build target user vector
    user_vector = pd.Series(0, index=mean_centered.columns, dtype=float)
    for m_id, r in st.session_state.user_ratings.items():
        if m_id in user_vector.index:
            user_vector[m_id] = r

    target_mean = user_vector[user_vector > 0].mean()
    user_vector = user_vector - target_mean
    user_vector = user_vector.fillna(0)

    # Cosine similarity
    similarities = cosine_similarity([user_vector], mean_centered.values)[0]

    # Overlap filter
    overlaps = (mean_centered != 0).dot((user_vector != 0).astype(int))
    valid_users = np.where(overlaps >= MIN_OVERLAP)[0]

    similarities_filtered = similarities[valid_users]
    top_idx = valid_users[np.argsort(similarities_filtered)[-20:]]

    # Weighted predictions
    preds = {}
    for movie_id in mean_centered.columns:
        if movie_id in st.session_state.user_ratings:
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
            pred = target_mean + (num / den)
            preds[movie_id] = pred

    top_movies = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:30]

    st.subheader("Recommended Movies")

    for m_id, _ in top_movies:
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
