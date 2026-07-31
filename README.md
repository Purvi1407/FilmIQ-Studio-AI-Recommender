# 🎥 FilmIQ Studio - AI Movie Recommendation System

An intelligent movie recommendation web app built using **Machine Learning** and **Streamlit**.
It suggests movies based on similarity of **genres, cast, keywords, and story patterns**.

---

## 🚀 Live Demo

👉 [https://filmiq-studio-ai-recommender-gjb4cw2r5omhh6ddul25m9.streamlit.app/](https://filmiq-studio-ai-recommender-gjb4cw2r5omhh6ddul25m9.streamlit.app/)

---

## 📌 Features

* 🎬 Content-based movie recommendation system
* 🤖 AI similarity using cosine similarity
* 📊 Text vectorization using CountVectorizer
* 🎯 Personalized recommendations
* 🎨 Beautiful interactive Streamlit UI
* ⚡ Fast response with caching optimization

---

## 🧠 How It Works

The system uses:

* Movie metadata (genres, cast, crew, keywords, overview)
* Text preprocessing and feature engineering
* Bag of Words model (CountVectorizer)
* Cosine similarity for ranking movies

### Workflow:

1. Combine movie features into tags
2. Convert text into vectors
3. Compute similarity matrix
4. Recommend top similar movies

---

## 🛠️ Tech Stack

* Python 🐍
* Streamlit ⚡
* Pandas 📊
* Scikit-learn 🤖
* NLP (Text Processing)

---

## 📂 Dataset

* TMDB 5000 Movies Dataset
* TMDB 5000 Credits Dataset

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/FilmIQ-Studio-AI-Recommender.git
cd FilmIQ-Studio-AI-Recommender
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📸 Screenshots

<img width="1898" height="868" alt="Screenshot 2026-05-16 200054" src="https://github.com/user-attachments/assets/b920b918-da4f-44db-8e47-c4eda31dd114" />
<img width="1906" height="867" alt="Screenshot 2026-05-16 200113" src="https://github.com/user-attachments/assets/f42ffd5b-d9a2-4fae-b433-0352da9fe861" />

---

## 🎯 Future Improvements

* 🎥 Add movie posters using TMDB API
* 🔍 Add search autocomplete
* 🎬 Trailer preview integration
* ⚡ Upgrade to FAISS for faster recommendations
* 🌐 Deploy on Streamlit Cloud / Render

---

## 👨‍💻 Author

Built by Purvi Lakhotia
Internship Project – Machine Learning + Streamlit

---

## ⭐ Show Support

If you like this project, give it a ⭐ on GitHub!

---
