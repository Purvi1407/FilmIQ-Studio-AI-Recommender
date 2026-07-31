# 🎥 FilmIQ Studio Pro - AI Movie Recommendation System

An intelligent **AI-powered movie recommendation platform** built using **Sentence-BERT (SBERT), Machine Learning, NLP, and Streamlit**.

FilmIQ Studio Pro uses **semantic understanding instead of simple keyword matching** to recommend movies. It generates contextual movie embeddings using **SBERT** and finds similar movies using **K-Nearest Neighbors (KNN)**.

The application also integrates the **TMDB API** for movie posters and implements optimized caching strategies for faster performance.

---

## 🚀 Live Demo

👉 https://filmiq-studio-ai-recommender-6ulpglpdevsgydvbvufavn.streamlit.app/

---

# ✨ Features

## 🤖 AI Recommendation Engine

* Semantic movie similarity using **SBERT embeddings**
* KNN-based nearest movie search
* Context-aware recommendations using movie metadata

## 🎬 Movie Discovery

* Search movies and get personalized recommendations
* Browse movies based on different moods:

  * 😊 Feel Good
  * 🔥 Intense
  * 🕯️ Dark
  * 😂 Funny
  * 🗺️ Adventurous
  * 🌙 Chill

## 🎥 TMDB Integration

* Dynamic movie poster fetching using TMDB API
* Lazy poster loading (fetches only displayed movies)
* Disk-based poster caching to reduce API calls

## 📌 User Features

* Add movies to personal watchlist
* Remove saved movies
* Export recommendations as CSV

## 📊 Analytics Dashboard

* Genre distribution analysis
* Movie rating visualization
* Dataset insights

## ⚡ Performance Optimization

* Streamlit caching for data and models
* Cached SBERT embeddings
* Optimized API requests using controlled thread pooling

---

# 🧠 How It Works

FilmIQ Studio Pro follows a **semantic content-based filtering approach**.

Unlike traditional recommendation systems that depend only on keyword similarity, SBERT understands the meaning and context of movie descriptions.

## Workflow

1. Load TMDB movie and credits datasets

2. Extract important movie information:

   * Movie overview
   * Genres
   * Cast
   * Director
   * Keywords

3. Combine metadata into a single text representation

4. Generate semantic embeddings using:

```
Sentence-BERT
(paraphrase-MiniLM-L3-v2)
```

5. Store movie vectors

6. Apply:

```
K-Nearest Neighbors
(Cosine Distance)
```

to find the most similar movies

7. Fetch movie posters from TMDB API

8. Display recommendations through Streamlit UI

---

# 🏗️ System Architecture

```
              User Input
                  |
                  ↓
        Movie Metadata Processing
                  |
                  ↓
        SBERT Embedding Generation
                  |
                  ↓
        Movie Vector Representation
                  |
                  ↓
        KNN Similarity Search
                  |
                  ↓
       Similar Movie Recommendations
                  |
                  ↓
          TMDB Poster API
                  |
                  ↓
        Streamlit Web Application
```

---

# 🛠️ Tech Stack

## Programming Language

* Python 🐍

## Machine Learning / NLP

* Sentence Transformers
* SBERT
* K-Nearest Neighbors
* Semantic Embeddings
* Natural Language Processing

## Data Processing

* Pandas
* NumPy

## Web Application

* Streamlit

## Visualization

* Plotly

## API Integration

* TMDB API

## Model Storage

* Joblib

---

# 📂 Dataset

The project uses:

* TMDB 5000 Movies Dataset
* TMDB 5000 Credits Dataset

Features used:

* Title
* Overview
* Genres
* Cast
* Director
* Keywords
* Ratings

---

# 📦 Installation

```bash
git clone https://github.com/yourusername/FilmIQ-Studio-Pro.git

cd FilmIQ-Studio-Pro

pip install -r requirements.txt
```

---

# 🔑 TMDB API Setup

Create a TMDB API key and add it inside:

```
.streamlit/secrets.toml
```

Example:

```toml
TMDB_API_KEY = "your_api_key_here"
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 📸 Screenshots

(Add your latest FilmIQ Studio Pro screenshots here)

---

# 🔮 Future Improvements

* 🎞️ Add movie trailer integration
* ⭐ Add user rating-based recommendations
* 🧠 Implement hybrid recommendation system  
  (Content + Collaborative Filtering)
* ⚡ Replace KNN search with FAISS for large-scale similarity search
* 🔍 Add intelligent movie search autocomplete
* 👥 Add user profiles and recommendation history
* 😊 Add facial emotion recognition to understand user mood and provide personalized movie recommendations based on detected emotions

---

# 👨‍💻 Author

Purvi Lakhotia

Machine Learning + NLP Internship Project

---

# ⭐ Show Support

If you like this project, consider giving it a ⭐ on GitHub!
