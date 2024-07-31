import pickle
import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=e84f821aae36bb4b01f4291f2cfe27f2&language=en-US"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    poster_path = data.get('poster_path')
    if poster_path:
        full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
        return full_path
    return None

def recommend(movie, movies_df, similarity_matrix):
    if movie not in movies_df['title'].values:
        return [], []

    index = movies_df[movies_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])

    recommended_movie_names = []
    recommended_movie_posters = []

    progress = st.progress(0)
    for i, (idx, similarity_score) in enumerate(distances[1:21]):  # Adjusted range to recommend 20 movies
        movie_id = movies_df.iloc[idx].id
        poster = fetch_poster(movie_id)
        if poster:
            recommended_movie_posters.append(poster)
            recommended_movie_names.append(movies_df.iloc[idx].title)
        progress.progress((i + 1) / 20)
        time.sleep(0.1)

        if len(recommended_movie_names) == 20:
            break
    progress.empty()
    return recommended_movie_names, recommended_movie_posters

st.set_page_config(page_title="Group-T", layout='wide')
st.markdown(
    """
    <style>
    .main {
        background-color: #3d3d3d;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .header {
        font-size: 40px;
        color: #4CAF50;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .movie-title {
        font-size: 16px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<h1 class='header'>Movie Recommendation systerm</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    with open('hybrid_movie_recommendation_model.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('similarity_movie_recommendation_model.pkl', 'rb') as f:
        similarity = pickle.load(f)

    if isinstance(data, dict) and 'movies' in data:
        movies = pd.DataFrame(data['movies'])
    elif isinstance(data, pd.DataFrame):
        movies = data
    else:
        st.error("Unexpected data format in hybrid_movie_recommendation_model.pkl")
        st.stop()

    if isinstance(similarity, dict) and 'content_sim' in similarity:
        similarity_matrix = np.array(similarity['content_sim'])
    elif isinstance(similarity, np.ndarray):
        similarity_matrix = similarity
    else:
        st.error("Unexpected data format in similarity_movie_recommendation_model.pkl")
        st.stop()

    return movies, similarity_matrix

movies, similarity = load_data()

if 'title' not in movies.columns:
    st.error("The 'title' column is missing from the movies DataFrame.")
    st.stop()

if 'id' not in movies.columns:
    st.error("The 'id' column is missing from the movies DataFrame.")
    st.stop()

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie",
    movie_list
)

if st.button('Show Recommendation'):
    with st.spinner('Fetching recommendations...'):
        recommended_movie_names, recommended_movie_posters = recommend(selected_movie, movies, similarity)

    if recommended_movie_names and recommended_movie_posters:
        rows = 4
        cols = 5
        grid = [st.columns(cols) for _ in range(rows)]

        for i, (name, poster) in enumerate(zip(recommended_movie_names, recommended_movie_posters)):
            row = i // cols
            col = i % cols
            with grid[row][col]:
                st.markdown(f"<p class='movie-title'>{name}</p>", unsafe_allow_html=True)
                st.image(poster)
    else:
        st.warning("No recommendations found for the selected movie.")
