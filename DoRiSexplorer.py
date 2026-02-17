import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.metrics.pairwise import cosine_similarity as cos_sim 
import streamlit as st
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
# st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the data
books = pd.read_csv('books.csv')
book_tags = pd.read_csv('book_tags.csv')
tags = pd.read_csv('tags.csv')
ratings = pd.read_csv('ratings.csv')

# Set page title
st.set_page_config(page_title="Book Explorer")

# Create a sidebar with options
st.sidebar.title("Explore the Book Data")
option = st.sidebar.selectbox(
    "Select an option", 
    ("Books", "Book Tags", "Ratings", "Get cosine similarities of top 100 raters", "Get cosine similarities of top N raters")
)

if option == "Books":
    st.header("Books Data")
    st.dataframe(books)

if option == "Book Tags":
    top_tags = book_tags.groupby('tag_id').count().sort_values('count', ascending=False).head(50)
    top_tags.reset_index(inplace=True)
    top_tags['tag_name'] = top_tags['tag_id'].apply(lambda x: tags.loc[tags['tag_id']==x, 'tag_name'].values[0])
    chart_data = top_tags[['tag_name', 'count']]
    bars = alt.Chart(chart_data).mark_bar().encode(
        x='count',
        y=alt.Y('tag_name', sort='-x'),
        tooltip=['tag_name', 'count']
    ).properties(title='Top 50 Most Popular Book Tags')
    st.altair_chart(bars, use_container_width=True)

if option == "Ratings":
    # Distribution of average ratings
    chart_data = books[['average_rating']]
    hist = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X("average_rating", bin=alt.Bin(maxbins=50)),
        y='count()',
        tooltip='count()'
    ).properties(title='Distribution of Average Ratings')
    st.altair_chart(hist, use_container_width=True)

    # Distribution of ratings
    chart_data = ratings['rating'].value_counts().sort_index().reset_index()
    chart_data.columns = ['Rating', 'Count']
    pie_chart = alt.Chart(chart_data).mark_arc().encode(
        theta='Count',
        color='Rating',
        tooltip=['Rating', 'Count']
    ).properties(title='Distribution of Ratings')
    st.altair_chart(pie_chart, use_container_width=True)

    # Scatter: number of ratings per user
    rating_counts = ratings['user_id'].value_counts()
    scatter_data = pd.DataFrame({'User': rating_counts.index, 'Rating Count': rating_counts.values})
    scatterplot = alt.Chart(scatter_data).mark_circle().encode(
        x='User',
        y='Rating Count',
        tooltip=['User', 'Rating Count']
    ).properties(title='Number of Times Raters Have Rated')
    st.altair_chart(scatterplot, use_container_width=True)

if option == "Get cosine similarities of top 100 raters":
    top_users = ratings['user_id'].value_counts().index[:100]
    ratings_top = ratings[ratings['user_id'].isin(top_users)]
    user_book_ratings = pd.pivot_table(ratings_top, values='rating', index='user_id', columns='book_id').fillna(0)
    user_book_ratings_sparse = sk.preprocessing.normalize(user_book_ratings, axis=0)
    sim = cos_sim(user_book_ratings_sparse)
    sim_df = pd.DataFrame(sim)
    sns.heatmap(sim_df, cmap="YlGnBu", xticklabels=False, yticklabels=False)
    plt.title("Cosine Similarity Heatmap of top 100 raters")
    st.pyplot()

if option == "Get cosine similarities of top N raters":
    max_users = min(943, len(ratings['user_id'].value_counts()))  # cap at dataset size
    top_n = st.slider("Number of top raters to include", min_value=10, max_value=max_users, value=100, step=10)
    top_users = ratings['user_id'].value_counts().index[:top_n]
    ratings_top = ratings[ratings['user_id'].isin(top_users)]
    user_book_ratings = pd.pivot_table(ratings_top, values='rating', index='user_id', columns='book_id').fillna(0)
    user_book_ratings_sparse = sk.preprocessing.normalize(user_book_ratings, axis=0)
    sim = cos_sim(user_book_ratings_sparse)
    sim_df = pd.DataFrame(sim)
    sns.heatmap(sim_df, cmap="YlGnBu", xticklabels=False, yticklabels=False)
    plt.title(f"Cosine Similarity Heatmap of top {top_n} raters")
    st.pyplot()
