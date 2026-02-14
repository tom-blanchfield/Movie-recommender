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
            return f"https://image.tmdb.org/t/p/w200{poster_path}"
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

# ---------- TOP 50% USERS ----------
user_counts = ratings["userId"].value_counts()
top_users = user_counts.head(int(len(user_counts) * 0.50)).index
ratin
