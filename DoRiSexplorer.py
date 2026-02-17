import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.metrics.pairwise import cosine_similarity as cos_sim 
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
# st.set_option('deprecation.showPyplotGlobalUse', False)

# Load ratings data
ratings = pd.read_csv('ratings.csv')

# Set page title
st.set_page_config(page_title="MoRiS Explorer")

# Sidebar options
st.sidebar.title("Explore the Movie Data")
option = st.sidebar.selectbox(
    "Select an option", 
    ("Ratings", "Get cosine similarities of top 100 raters", "Get cosine similarities of top N raters")
)

if option == "Ratings":
    # Distribution of ratings
    chart_data = ratings['rating'].value_counts().sort_index().reset_index()
    chart_data.columns = ['Rating', 'Count']
    pie_chart = chart_data.plot.pie(y='Count', labels=chart_data['Rating'], autopct='%1.1f%%', legend=False, figsize=(5,5))
    st.pyplot(pie_chart.figure)
    
    # Scatter: number of ratings per user
    rating_counts = ratings['userId'].value_counts()
    scatter_data = pd.DataFrame({'User': rating_counts.index, 'Rating Count': rating_counts.values})
    plt.figure(figsize=(8,4))
    plt.scatter(scatter_data['User'], scatter_data['Rating Count'])
    plt.xlabel("User ID")
    plt.ylabel("Number of Ratings")
    plt.title("Number of Ratings per User")
    st.pyplot(plt)

if option == "Get cosine similarities of top 100 raters":
    top_users = ratings['userId'].value_counts().index[:100]
    ratings_top = ratings[ratings['userId'].isin(top_users)]
    user_movie_matrix = pd.pivot_table(ratings_top, values='rating', index='userId', columns='movieId').fillna(0)
    user_movie_matrix_sparse = sk.preprocessing.normalize(user_movie_matrix, axis=0)
    sim = cos_sim(user_movie_matrix_sparse)
    sim_df = pd.DataFrame(sim)
    plt.figure(figsize=(8,6))
    sns.heatmap(sim_df, cmap="YlGnBu", xticklabels=False, yticklabels=False)
    plt.title("Cosine Similarity Heatmap of top 100 raters")
    st.pyplot()

if option == "Get cosine similarities of top N raters":
    max_users = min(943, len(ratings['userId'].value_counts()))  # cap at dataset size
    top_n = st.slider("Number of top raters to include", min_value=10, max_value=max_users, value=100, step=10)
    top_users = ratings['userId'].value_counts().index[:top_n]
    ratings_top = ratings[ratings['userId'].isin(top_users)]
    user_movie_matrix = pd.pivot_table(ratings_top, values='rating', index='userId', columns='movieId').fillna(0)
    user_movie_matrix_sparse = sk.preprocessing.normalize(user_movie_matrix, axis=0)
    sim = cos_sim(user_movie_matrix_sparse)
    sim_df = pd.DataFrame(sim)
    plt.figure(figsize=(8,6))
    sns.heatmap(sim_df, cmap="YlGnBu", xticklabels=False, yticklabels=False)
    plt.title(f"Cosine Similarity Heatmap of top {top_n} raters")
    st.pyplot()
