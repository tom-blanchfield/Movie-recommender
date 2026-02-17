import pandas as pd
import numpy as np
import streamlit as st
import sklearn as sk
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from PIL import Image
import requests

st.set_page_config(page_title="DoRiS Explorer", layout="wide")

# -------------------- DATA SELECTION --------------------
dataset = st.sidebar.selectbox("Choose dataset", ["Movies", "Books"])

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_movies():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    links = pd.read_csv("links.csv")
    movies["genres"] = movies["genres"].fillna("")
    movies = movies.merge(links, on="movieId", how="left")
    rating_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    rating_stats.columns = ["movieId", "avg_rating", "rating_count"]
    movies = movies.merge(rating_stats, on="movieId", how="left")
    return ratings, movies

@st.cache_data
def load_books():
    books = pd.read_csv("books.csv")
    book_tags = pd.read_csv("book_tags.csv")
    tags = pd.read_csv("tags.csv")
    ratings = pd.read_csv("ratings.csv")
    book_data = pd.merge(books, book_tags, on='goodreads_book_id', how='left')
    book_data = pd.merge(book_data, tags, on='tag_id', how='left')
    return ratings, books, book_data

if dataset == "Movies":
    ratings, movies = load_movies()
else:
    ratings, books, book_data = load_books()

# -------------------- OPTIONS --------------------
st.sidebar.title("Explorer Options")
option = st.sidebar.selectbox("Select view", ["Data Table", "Ratings Distribution", "Top Raters Cosine Similarity"])

# -------------------- MOVIE EXPLORER --------------------
if dataset == "Movies":
    if option == "Data Table":
        st.header("Movies Data")
        st.dataframe(movies)
        st.header("Ratings Data")
        st.dataframe(ratings)

    elif option == "Ratings Distribution":
        st.header("Distribution of Ratings")
        hist = alt.Chart(ratings).mark_bar().encode(
            x=alt.X("rating", bin=alt.Bin(maxbins=50)),
            y='count()',
            tooltip='count()'
        ).properties(title='Ratings Distribution')
        st.altair_chart(hist, use_container_width=True)

        avg_ratings = movies[['title','avg_rating']]
        st.header("Distribution of Average Ratings")
        hist2 = alt.Chart(avg_ratings).mark_bar().encode(
            x=alt.X("avg_rating", bin=alt.Bin(maxbins=50)),
            y='count()',
            tooltip='count()'
        ).properties(title='Avg Movie Ratings Distribution')
        st.altair_chart(hist2, use_container_width=True)

    elif option == "Top Raters Cosine Similarity":
        top_n = st.sidebar.slider("Top N raters", min_value=50, max_value=2000, value=500, step=50)
        top_users = ratings['userId'].value_counts().index[:top_n]
        ratings_top = ratings[ratings['userId'].isin(top_users)]
        user_movie_matrix = ratings_top.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
        user_norm = sk.preprocessing.normalize(user_movie_matrix, axis=1)
        sim = cos_sim(user_norm)
        st.header(f"Cosine Similarity Heatmap for top {top_n} movie raters")
        sns.heatmap(sim, cmap="YlGnBu", xticklabels=False, yticklabels=False)
        st.pyplot()

# -------------------- BOOK EXPLORER --------------------
else:
    if option == "Data Table":
        st.header("Books Data")
        st.dataframe(books)
        st.header("Book Tags")
        st.dataframe(book_data[['title','tag_name','count']])
        st.header("Ratings Data")
        st.dataframe(ratings)

    elif option == "Ratings Distribution":
        st.header("Distribution of Ratings")
        hist = alt.Chart(ratings).mark_bar().encode(
            x=alt.X("rating", bin=alt.Bin(maxbins=50)),
            y='count()',
            tooltip='count()'
        ).properties(title='Ratings Distribution')
        st.altair_chart(hist, use_container_width=True)

        avg_ratings = books[['title','average_rating']]
        st.header("Distribution of Average Ratings")
        hist2 = alt.Chart(avg_ratings).mark_bar().encode(
            x=alt.X("average_rating", bin=alt.Bin(maxbins=50)),
            y='count()',
            tooltip='count()'
        ).properties(title='Avg Book Ratings Distribution')
        st.altair_chart(hist2, use_container_width=True)

    elif option == "Top Raters Cosine Similarity":
        top_n = st.sidebar.slider("Top N raters", min_value=50, max_value=2000, value=500, step=50)
        top_users = ratings['user_id'].value_counts().index[:top_n]
        ratings_top = ratings[ratings['user_id'].isin(top_users)]
        user_book_matrix = ratings_top.pivot_table(index='user_id', columns='book_id', values='rating').fillna(0)
        user_norm = sk.preprocessing.normalize(user_book_matrix, axis=1)
        sim = cos_sim(user_norm)
        st.header(f"Cosine Similarity Heatmap for top {top_n} book raters")
        sns.heatmap(sim, cmap="YlGnBu", xticklabels=False, yticklabels=False)
        st.pyplot()
