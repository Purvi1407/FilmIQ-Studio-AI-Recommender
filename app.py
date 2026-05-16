import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="FilmIQ Studio",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CSS (UNCHANGED - YOUR FULL DESIGN)
# =========================================================

st.markdown("""
<style>

/* APP BACKGROUND */
[data-testid="stAppViewContainer"] {
    background:
    linear-gradient(rgba(0,0,0,0.22), rgba(0,0,0,0.22)),
    url("https://images.unsplash.com/photo-1524985069026-dd778a71c7b4?auto=format&fit=crop&w=2400&q=100");

    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: fixed;

    width: 100%;
    min-height: 100vh;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: rgba(10, 15, 30, 0.72) !important;
    backdrop-filter: blur(14px);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* TITLE */
.main-title {
    text-align: center;
    font-size: 82px;
    font-weight: 800;
    color: #6ee7ff;
    margin-top: 20px;
    text-shadow: 0px 0px 20px rgba(255,75,75,0.6),
                 2px 2px 15px rgba(0,0,0,0.9);
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    color: #ffffff;
    font-size: 23px;
    line-height: 1.9;
    margin-bottom: 50px;
}

/* CARD */
.card {
    background: rgba(255, 255, 255, 0.06);
    padding: 42px;
    border-radius: 28px;
    margin-bottom: 30px;
    border: 1px solid rgba(110, 231, 255, 0.18);
    box-shadow: 0px 10px 35px rgba(0,0,0,0.4);
    backdrop-filter: blur(14px);
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 14px;
    padding: 15px 28px;
    font-size: 18px;
    font-weight: 700;
    width: 100%;
}

/* MOVIE CARD */
.movie-card {
    background: rgba(15,15,15,0.72);
    padding: 24px;
    border-radius: 18px;
    margin-bottom: 18px;
    border-left: 5px solid #6ee7ff;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #f0f0f0;
    margin-top: 45px;
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎬 CineMatch AI")
st.sidebar.markdown("---")
st.sidebar.write("🎯 AI Recommendation Engine")
st.sidebar.success("System Active ✅")

# =========================================================
# 🚀 FULL MODEL (FAST CACHE FIX)
# =========================================================

@st.cache_data
def build_model():

    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(credits, on='title')

    movies = movies[
        ['movie_id', 'title', 'overview',
         'genres', 'keywords', 'cast', 'crew']
    ]

    movies.dropna(inplace=True)

    def convert(text):
        return [i['name'] for i in ast.literal_eval(text)]

    def convert_cast(text):
        return [i['name'] for i in ast.literal_eval(text)[:3]]

    def fetch_director(text):
        return [i['name'] for i in ast.literal_eval(text) if i['job'] == 'Director']

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(fetch_director)

    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

    movies['tags'] = (
        movies['overview'] +
        movies['genres'] +
        movies['keywords'] +
        movies['cast'] +
        movies['crew']
    )

    new_df = movies[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()

    similarity = cosine_similarity(vectors)

    return new_df, similarity


# =========================================================
# LOAD MODEL ONCE
# =========================================================

new_df, similarity = build_model()

# =========================================================
# RECOMMEND FUNCTION
# =========================================================

def recommend(movie):

    if movie not in new_df['title'].values:
        return []

    idx = new_df[new_df['title'] == movie].index[0]
    distances = similarity[idx]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    return [new_df.iloc[i[0]].title for i in movies_list]

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class='main-title'>🎥 FilmIQ Studio</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='subtitle'>
AI Movie Recommendation System using Machine Learning
</div>
""", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class='card'>
<h2 style='text-align:center; color:white;'>🍿 Discover Your Next Favorite Movie</h2>
</div>
""", unsafe_allow_html=True)

# =========================================================
# UI INPUTS
# =========================================================

st.markdown("### 🎯 Smart Mood Selector")

mood = st.selectbox(
    "What are you in the mood for?",
    ["🎭 Drama", "😂 Comedy", "💥 Action", "❤️ Romance", "🧠 Mind-bending"]
)

st.subheader("🎥 Select a Movie")

selected_movie = st.selectbox(
    "Choose a movie",
    ["Choose a movie"] + list(new_df['title'].values),
    label_visibility="collapsed"
)

# =========================================================
# BUTTON ACTION
# =========================================================

if st.button("🚀 Generate Recommendations"):

    if selected_movie == "Choose a movie":
        st.warning("⚠️ Please select a movie first")

    else:
        with st.spinner("Finding best matches... ⚡"):
            results = recommend(selected_movie)

        st.subheader("✨ Recommended For You")

        for movie in results:
            st.markdown(f"""
            <div class="movie-card">
                <h3>🎬 {movie}</h3>
                <p style='color:white;'>
                AI-based similarity recommendation
                </p>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class='footer'>
Built with ❤️ using Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)
