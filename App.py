import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Collaborative Movie Recommender")

# ---------- TMDB API ----------
TMDB_API_KEY = "888bb40cd1f4d3c95b375753e9c34c09"

@st.cache_data
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
    except:
        return None
    return None

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    tags = pd.read_csv("tags.csv")
    links = pd.read_csv("links.csv")
    return ratings, movies, tags, links

ratings, movies, tags, links = load_data()
movies["genres"] = mov
