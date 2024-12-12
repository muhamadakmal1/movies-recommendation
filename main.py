import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load movie data
def load_data():
    movies = pd.read_csv("movies.csv")  # Replace with your dataset path
    return movies

def recommend_movies(movie_title, movies, tfidf_matrix):
    try:
        # Get index of the movie
        movie_idx = movies[movies['title'].str.lower() == movie_title.lower()].index[0]

        # Compute cosine similarity
        cosine_similarities = linear_kernel(tfidf_matrix[movie_idx], tfidf_matrix).flatten()

        # Get top 10 most similar movies
        similar_indices = cosine_similarities.argsort()[-11:-1][::-1]

        recommended_movies = movies.iloc[similar_indices]
        return recommended_movies
    except IndexError:
        return None

# Streamlit app
def main():
    st.title("Movie Recommendation System")

    st.sidebar.header("Settings")
    
    # Load movie data
    st.text("Loading data...")
    movies = load_data()

    # Preprocess data
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    movies['overview'] = movies['overview'].fillna('')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies['overview'])

    # User input
    movie_title = st.text_input("Enter a movie title you like:", "")

    if st.button("Recommend"):
        if movie_title:
            recommended_movies = recommend_movies(movie_title, movies, tfidf_matrix)
            if recommended_movies is not None:
                st.subheader("Recommended Movies:")
                for _, row in recommended_movies.iterrows():
                    st.write(f"- {row['title']}")
            else:
                st.error("Movie not found. Please check the title and try again.")
        else:
            st.error("Please enter a movie title.")

if __name__ == "__main__":
    main()
