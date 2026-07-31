# FilmIQ Studio Pro - SBERT + KNN Movie Recommendation System
# Place tmdb_5000_movies.csv and tmdb_5000_credits.csv in the same folder.
# Also place .streamlit/config.toml and bg_image_data.py alongside this file.
# Run: streamlit run app.py

import os
import ast
import time
import random
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from bg_image_data import BG_DATA_URI  # background image, kept in its own
                                        # file since it's ~140k chars of
                                        # base64 — far too long to inline
                                        # safely inside this file's CSS string

st.set_page_config(page_title="FilmIQ Studio Pro", page_icon="🎬", layout="wide")

CACHE_FILE = "model_cache.joblib"
POSTER_CACHE_FILE = "poster_cache.joblib"  # disk cache: {tmdb_id: poster_url}

# --- TMDB API key ---------------------------------------------------------
# Get a free key at https://www.themoviedb.org/settings/api
# Put it in .streamlit/secrets.toml as: TMDB_API_KEY = "your_key_here"
try:
    _secret_key = st.secrets.get("TMDB_API_KEY", "")
except Exception:
    _secret_key = ""

TMDB_API_KEY = os.environ.get("TMDB_API_KEY") or _secret_key or ""
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_POSTER = "https://placehold.co/500x750/1a1a1e/888888?text=No+Poster"


# ---------------------------------------------------------------------------
# Mood definitions — maps a mood to genres it favors, genres it avoids,
# and a minimum rating bar. This is a heuristic layer on top of the
# dataset since TMDB has no native "mood" field.
# ---------------------------------------------------------------------------
MOODS = {
    "Feel Good": {
        "favor": ["Comedy", "Family", "Animation", "Music", "Romance"],
        "avoid": ["Horror", "War", "Crime"],
        "min_rating": 6.0,
        "emoji": "😊",
    },
    "Intense": {
        "favor": ["Thriller", "Action", "Crime", "War"],
        "avoid": ["Family", "Animation"],
        "min_rating": 0.0,
        "emoji": "🔥",
    },
    "Dark": {
        "favor": ["Horror", "Mystery", "Thriller", "Crime"],
        "avoid": ["Comedy", "Family", "Animation"],
        "min_rating": 0.0,
        "emoji": "🕯️",
    },
    "Funny": {
        "favor": ["Comedy"],
        "avoid": ["Horror", "War"],
        "min_rating": 0.0,
        "emoji": "😂",
    },
    "Adventurous": {
        "favor": ["Adventure", "Fantasy", "Action", "Science Fiction"],
        "avoid": [],
        "min_rating": 0.0,
        "emoji": "🗺️",
    },
    "Chill": {
        "favor": ["Drama", "Romance", "Documentary", "Music"],
        "avoid": ["Horror", "War", "Action"],
        "min_rating": 0.0,
        "emoji": "🌙",
    },
}


# ---------------------------------------------------------------------------
# Poster fetching — calls TMDB's API by movie id and caches results to
# disk so we only ever hit the network once per movie, not once per
# rerun. Without this, every script rerun would re-fetch 5000 posters.
# ---------------------------------------------------------------------------
def _load_poster_cache():
    if os.path.exists(POSTER_CACHE_FILE):
        try:
            return joblib.load(POSTER_CACHE_FILE)
        except Exception:
            return {}
    return {}


def _save_poster_cache(cache):
    try:
        joblib.dump(cache, POSTER_CACHE_FILE)
    except Exception:
        pass


def _fetch_one_poster(session, movie_id):
    """Fetches a single movie's poster URL. Always returns a usable
    string (never None) so callers can render <img src="..."> safely."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    try:
        response = session.get(url, params={"api_key": TMDB_API_KEY}, timeout=10)

        if response.status_code == 429:
            time.sleep(1)
            response = session.get(url, params={"api_key": TMDB_API_KEY}, timeout=10)

        if response.status_code == 200:
            poster_path = response.json().get("poster_path")
            return movie_id, (f"{TMDB_IMG_BASE}{poster_path}" if poster_path else PLACEHOLDER_POSTER)

        return movie_id, PLACEHOLDER_POSTER

    except requests.RequestException:
        return movie_id, PLACEHOLDER_POSTER


def fetch_posters(tmdb_ids, progress_callback=None, max_workers=20):
    """
    Given a list of TMDB movie ids, returns {id: poster_url}.
    Fetches missing ids concurrently (thread pool) instead of one at a
    time — this is the main speed fix, since ~5000 sequential HTTP
    round-trips is what made the first run painfully slow.
    Every value is guaranteed to be a real URL string — never None.
    """
    cache = _load_poster_cache()

    valid_ids = []
    for movie_id in tmdb_ids:
        try:
            movie_id = int(movie_id)
            if movie_id > 0:
                valid_ids.append(movie_id)
        except (TypeError, ValueError):
            continue

    missing = [i for i in valid_ids if i not in cache]

    if not missing:
        return cache

    if not TMDB_API_KEY:
        for movie_id in missing:
            cache[movie_id] = PLACEHOLDER_POSTER
        return cache

    session = requests.Session()
    total = len(missing)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one_poster, session, mid): mid for mid in missing}

        for future in as_completed(futures):
            movie_id, poster_url = future.result()
            cache[movie_id] = poster_url
            completed += 1

            if progress_callback and completed % 25 == 0:
                progress_callback(completed / total)

            # Save incrementally every ~250 completions so an interrupt
            # doesn't lose all progress and force a full re-fetch.
            if completed % 250 == 0:
                _save_poster_cache(cache)

    _save_poster_cache(cache)
    return cache


# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    credits.rename(columns={"movie_id": "id"}, inplace=True)
    df = movies.merge(credits, on="id")

    def parse_names(x, limit=None):
        try:
            data = ast.literal_eval(x)
            names = [i["name"] for i in data]
            return names[:limit] if limit else names
        except (ValueError, SyntaxError, TypeError):
            return []

    def director(x):
        try:
            crew = ast.literal_eval(x)
            for c in crew:
                if c.get("job") == "Director":
                    return c.get("name", "")
        except (ValueError, SyntaxError, TypeError):
            pass
        return ""

    df["genres_list"] = df["genres"].apply(parse_names)
    df["keywords_list"] = df["keywords"].apply(lambda x: parse_names(x, 8))
    df["cast_list"] = df["cast"].apply(lambda x: parse_names(x, 5))
    df["director"] = df["crew"].apply(director)

    df["combined_text"] = (
        df["overview"].fillna("") + " " +
        df["genres_list"].apply(lambda x: " ".join(x)) + " " +
        df["cast_list"].apply(lambda x: " ".join(x)) + " " +
        df["director"].fillna("") + " " +
        df["keywords_list"].apply(lambda x: " ".join(x))
    )

    df = df[df["combined_text"].str.strip() != ""].reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def attach_posters(df, _poster_signature):
    """
    Fetches/loads poster URLs for every movie in df and attaches them
    as df["poster_url"]. Cached by Streamlit so this whole function
    (including the cache-file read and the per-row .apply) only runs
    once per session instead of on every single rerun/button click.
    _poster_signature is a cheap proxy so the cache still invalidates
    if the underlying id list changes.
    """
    poster_cache = fetch_posters(df["id"].tolist())
    df = df.copy()
    df["poster_url"] = df["id"].apply(
        lambda i: poster_cache.get(int(i), PLACEHOLDER_POSTER) if pd.notna(i) else PLACEHOLDER_POSTER
    )
    return df


def attach_posters_with_progress(df, _poster_signature, progress_bar):
    """
    Same as attach_posters but drives a live progress bar — used only
    on the very first run (cold cache) since that's the only time the
    fetch takes long enough to matter. Subsequent calls hit the
    @st.cache_data-wrapped attach_posters above and return instantly.
    """
    def _update(frac):
        progress_bar.progress(frac, text=f"Fetching posters from TMDB... {int(frac * 100)}%")

    poster_cache = fetch_posters(df["id"].tolist(), progress_callback=_update)
    df = df.copy()
    df["poster_url"] = df["id"].apply(
        lambda i: poster_cache.get(int(i), PLACEHOLDER_POSTER) if pd.notna(i) else PLACEHOLDER_POSTER
    )
    return df


# ---------------------------------------------------------------------------
# Model / embedding build — cached to disk so the heavy encode step
# only ever runs once.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_model(_df, signature):
    if os.path.exists(CACHE_FILE):
        cached = joblib.load(CACHE_FILE)
        if cached.get("signature") == signature:
            return cached["embeddings"], cached["knn"], cached["idx_map"]

    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

    embeddings = model.encode(
        _df["combined_text"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=64,
        normalize_embeddings=True,
    )

    n_neighbors = min(50, len(_df))
    knn = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors)
    knn.fit(embeddings)

    idx_map = pd.Series(
        _df.index,
        index=_df["title_x"].str.lower(),
    ).drop_duplicates()

    joblib.dump(
        {"embeddings": embeddings, "knn": knn, "idx_map": idx_map, "signature": signature},
        CACHE_FILE,
    )

    return embeddings, knn, idx_map


def recommend(title, df, embeddings, knn, idx_map, n=10):
    key = title.lower()
    if key not in idx_map.index:
        return pd.DataFrame()

    idx = idx_map[key]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    vec = embeddings[idx].reshape(1, -1)
    n_query = min(n + 1, embeddings.shape[0])
    distances, indices = knn.kneighbors(vec, n_neighbors=n_query)

    rows = []
    for dist, i in zip(distances[0][1:], indices[0][1:]):
        row = df.iloc[i]
        rows.append({
            "id": row["id"],
            "Title": row["title_x"],
            "Similarity": round((1 - dist) * 100, 2),
            "Rating": row["vote_average"],
            "Director": row["director"],
            "Genres": ", ".join(row["genres_list"][:4]),
            "Overview": str(row["overview"])[:250],
            "Poster": row["poster_url"],
        })
    return pd.DataFrame(rows)


def filter_by_mood(df, mood_name):
    """Returns the subset of df matching a mood's genre/rating rules."""
    rule = MOODS[mood_name]
    favor, avoid, min_rating = rule["favor"], rule["avoid"], rule["min_rating"]

    def matches(genres):
        if avoid and any(g in avoid for g in genres):
            return False
        if favor and not any(g in favor for g in genres):
            return False
        return True

    mask = df["genres_list"].apply(matches) & (df["vote_average"] >= min_rating)
    result = df[mask]

    # Fallback: if a mood is too strict for this dataset and returns
    # nothing, relax to favor-only matching so the UI never dead-ends.
    if result.empty and favor:
        result = df[df["genres_list"].apply(lambda g: any(x in favor for x in g))]

    return result


# ---------------------------------------------------------------------------
# Styling — cinematic dark theme
#
# Token system:
#   bg canvas      #0a0a0c   near-black, slightly warm
#   surface        #16161a   card background
#   surface-raised #1e1e23   hovered/active surface
#   accent         #e0334d   warm red, primary actions + active states
#   gold           #f4b740   ratings, "in watchlist" affirmations
#   text primary   #f5f5f5
#   text muted     #8a8a93
#   hairline       #26262c
#
# Display type leans condensed/heavy (cinematic marquee feel) via the
# Bebas Neue / Oswald stack, set in small-caps with wide letter-spacing
# for eyebrows and tab labels — body text stays a clean system sans so
# overviews stay easy to read.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

_CSS_TEMPLATE = """
    <style>
    :root {
        --bg: #0a0a0c;
        --surface: #16161a;
        --surface-raised: #1e1e23;
        --accent: #e0334d;
        --accent-dim: #7a1c29;
        --gold: #f4b740;
        --text: #f5f5f5;
        --text-muted: #8a8a93;
        --hairline: #26262c;
    }

    .stApp {
        background-image:
            linear-gradient(to bottom, rgba(10,10,12,0.94) 0%, rgba(10,10,12,0.80) 45%, rgba(10,10,12,0.92) 100%),
            url("__BG_DATA_URI__");
        background-color: var(--bg);
        background-size: cover, cover;
        background-position: center, center;
        background-repeat: no-repeat, no-repeat;
        background-attachment: fixed, fixed;
    }
    section.main > div { padding-top: 1.2rem; }

    body, .stApp, p, span, label, div { font-family: 'Inter', -apple-system, sans-serif; }

    /* ---------- Hero header ---------- */
    .hero-wrap {
        text-align: center;
        padding: 28px 0 8px;
        margin-bottom: 8px;
        border-bottom: 1px solid var(--hairline);
    }
    .hero-eyebrow {
        font-family: 'Oswald', sans-serif;
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 0.28em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 6px;
    }
    .hero-title {
        font-family: 'Oswald', sans-serif;
        font-size: 52px;
        font-weight: 700;
        letter-spacing: 0.01em;
        color: var(--text);
        line-height: 1.05;
        margin: 0;
    }
    .hero-subtitle {
        color: var(--text-muted);
        font-size: 15px;
        margin-top: 8px;
        margin-bottom: 4px;
    }

    /* ---------- Tab bar (Streamlit native tabs, restyled) ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid var(--hairline);
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Oswald', sans-serif;
        font-size: 14px;
        letter-spacing: 0.04em;
        font-weight: 500;
        color: var(--text-muted);
        background-color: transparent;
        border-radius: 0;
        padding: 12px 20px;
    }
    .stTabs [aria-selected="true"] {
        color: var(--text) !important;
        border-bottom: 2px solid var(--accent) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent); }

    /* ---------- Movie card ---------- */
    .movie-card {
        background-color: var(--surface);
        border-radius: 6px;
        padding: 0;
        margin-bottom: 10px;
        overflow: hidden;
        border: 1px solid var(--hairline);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        position: relative;
    }
    .movie-card:hover {
        transform: translateY(-4px);
        border-color: var(--accent);
        box-shadow: 0 12px 24px -8px rgba(224, 51, 77, 0.35);
    }
    .movie-card .poster-wrap {
        position: relative;
        width: 100%;
        aspect-ratio: 2/3;
        background-color: var(--surface-raised);
        overflow: hidden;
    }
    .movie-card img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }
    .movie-card .poster-fade {
        position: absolute;
        left: 0; right: 0; bottom: 0;
        height: 50%;
        background: linear-gradient(to bottom, rgba(10,10,12,0) 0%, rgba(10,10,12,0.92) 100%);
        pointer-events: none;
    }
    .similarity-badge {
        position: absolute;
        top: 8px; right: 8px;
        background-color: rgba(10, 10, 12, 0.78);
        border: 1px solid var(--accent);
        color: var(--text);
        font-family: 'Oswald', sans-serif;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.02em;
        padding: 3px 9px;
        border-radius: 20px;
    }
    .movie-card .card-body { padding: 10px 12px 12px; }
    .movie-title {
        font-weight: 600;
        font-size: 14.5px;
        color: var(--text);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        line-height: 1.3;
    }
    .movie-meta {
        font-size: 12px;
        color: var(--text-muted);
        margin-top: 3px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .movie-meta .star { color: var(--gold); }

    /* ---------- Buttons: tighten and theme the watchlist toggle ---------- */
    div[data-testid="stButton"] button {
        border-radius: 5px;
        font-family: 'Oswald', sans-serif;
        font-size: 12.5px;
        letter-spacing: 0.03em;
        font-weight: 500;
        border: 1px solid var(--hairline);
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: var(--accent);
        border-color: var(--accent);
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #c92840;
        border-color: #c92840;
    }

    /* ---------- Section labels ---------- */
    .section-eyebrow {
        font-family: 'Oswald', sans-serif;
        font-size: 12px;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 2px;
    }
    .section-title {
        font-family: 'Oswald', sans-serif;
        font-size: 24px;
        font-weight: 600;
        color: var(--text);
        margin-top: 0;
        margin-bottom: 16px;
    }

    </style>
    """

st.markdown(
    _CSS_TEMPLATE.replace("__BG_DATA_URI__", BG_DATA_URI),
    unsafe_allow_html=True,
)


def _init_watchlist():
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = {}  # {id: row_dict}


def add_to_watchlist(row):
    _init_watchlist()
    movie_id = row.get("id")
    if movie_id is None or pd.isna(movie_id):
        return
    st.session_state.watchlist[int(movie_id)] = {
        "id": int(movie_id),
        "Title": row.get("Title"),
        "Rating": row.get("Rating"),
        "Genres": row.get("Genres"),
        "Overview": row.get("Overview"),
        "Poster": row.get("Poster"),
        "Director": row.get("Director", ""),
    }


def remove_from_watchlist(movie_id):
    _init_watchlist()
    st.session_state.watchlist.pop(int(movie_id), None)


def is_in_watchlist(movie_id):
    _init_watchlist()
    return int(movie_id) in st.session_state.watchlist if pd.notna(movie_id) else False


def render_movie_grid(rows_df, columns=5, show_similarity=False, key_prefix="grid"):
    """Renders a list of movie dicts/rows as poster cards in a grid,
    each with an Add/Remove Watchlist toggle button."""
    if rows_df.empty:
        st.info("No movies match right now — try a different mood or filter.")
        return

    _init_watchlist()
    cols = st.columns(columns)
    for i, (_, row) in enumerate(rows_df.iterrows()):
        with cols[i % columns]:
            poster = row["Poster"] if pd.notna(row.get("Poster")) else PLACEHOLDER_POSTER
            similarity_html = ""
            if show_similarity and "Similarity" in row:
                similarity_html = f'<div class="similarity-badge">{row["Similarity"]}% match</div>'

            st.markdown(
                f"""
                <div class="movie-card">
                    <div class="poster-wrap">
                        <img src="{poster}" onerror="this.onerror=null;this.src='{PLACEHOLDER_POSTER}';" />
                        <div class="poster-fade"></div>
                        {similarity_html}
                    </div>
                    <div class="card-body">
                        <div class="movie-title">{row['Title']}</div>
                        <div class="movie-meta"><span class="star">★</span> {row['Rating']:.1f} · {row['Genres']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            movie_id = row.get("id")
            has_id = movie_id is not None and pd.notna(movie_id)
            btn_key = f"{key_prefix}_wl_{int(movie_id) if has_id else i}"

            if has_id and is_in_watchlist(movie_id):
                if st.button("✓ In Watchlist", key=btn_key, use_container_width=True):
                    remove_from_watchlist(movie_id)
                    st.rerun()
            else:
                if st.button("+ Watchlist", key=btn_key, use_container_width=True, disabled=not has_id):
                    add_to_watchlist(row)
                    st.rerun()

            with st.expander("Overview"):
                st.write(row["Overview"])
                if "Director" in row and row["Director"]:
                    st.caption(f"Directed by {row['Director']}")


# ---------------------------------------------------------------------------
# Load data + posters + model
# ---------------------------------------------------------------------------
df = load_data()
poster_signature = (len(df), tuple(df["id"].head(20)))

if not TMDB_API_KEY:
    st.warning(
        "No TMDB_API_KEY found — showing placeholder posters. "
        "Add your key to .streamlit/secrets.toml as TMDB_API_KEY = \"your_key\" and rerun.",
        icon="🔑",
    )

if "posters_attached" not in st.session_state:
    progress = st.progress(0.0, text="Fetching posters from TMDB (first run only)...")
    df = attach_posters_with_progress(df, poster_signature, progress)
    progress.empty()
    st.session_state.posters_attached = True
else:
    # Cached by Streamlit — instant after the first run, no re-fetch,
    # no re-apply over the dataframe.
    df = attach_posters(df, poster_signature)

signature = (len(df), hash(tuple(df["title_x"].head(20))))

with st.spinner("Loading recommendation engine (first run only)..."):
    embeddings, knn, idx_map = build_model(df, signature)

if "surprise_pick" not in st.session_state:
    st.session_state.surprise_pick = None
if "active_mood" not in st.session_state:
    st.session_state.active_mood = None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-eyebrow">Semantic Recommendation Engine</div>
        <div class="hero-title">FILMIQ STUDIO PRO</div>
        <div class="hero-subtitle">Find your next watch by title, mood, or pure chance</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_search, tab_mood, tab_surprise, tab_watchlist, tab_analytics = st.tabs(
    ["🔍 Search by Movie", "🎭 Browse by Mood", "🎲 Surprise Me", "📌 Watchlist", "📊 Analytics"]
)


# ---------------------------------------------------------------------------
# Tab 1: classic title-based recommendation
# ---------------------------------------------------------------------------
with tab_search:
    col_a, col_b = st.columns([3, 1])
    with col_a:
        movie = st.selectbox("Pick a movie you like", sorted(df["title_x"].dropna().unique()))
    with col_b:
        top_n = st.slider("How many results", 5, 20, 10)

    if st.button("Recommend", type="primary"):
        recs = recommend(movie, df, embeddings, knn, idx_map, top_n)
        if recs.empty:
            st.warning(f"No match found for '{movie}'. Try a different title.")
        else:
            st.markdown(
                f"""
                <div class="section-eyebrow">Tailored picks</div>
                <div class="section-title">Because you liked {movie}</div>
                """,
                unsafe_allow_html=True,
            )
            render_movie_grid(recs, columns=5, show_similarity=True, key_prefix="search")
            st.download_button(
                "Download as CSV",
                recs.to_csv(index=False),
                "recommendations.csv",
                "text/csv",
            )


# ---------------------------------------------------------------------------
# Tab 2: mood-based browsing
# ---------------------------------------------------------------------------
with tab_mood:
    st.markdown(
        '<div class="section-eyebrow">Set the tone</div>'
        '<div class="section-title">What are you in the mood for?</div>',
        unsafe_allow_html=True,
    )

    mood_cols = st.columns(len(MOODS))
    for i, (mood_name, rule) in enumerate(MOODS.items()):
        with mood_cols[i]:
            is_active = st.session_state.active_mood == mood_name
            label = f"{rule['emoji']} {mood_name}" + (" ✓" if is_active else "")
            if st.button(
                label,
                use_container_width=True,
                key=f"mood_{mood_name}",
                type="primary" if is_active else "secondary",
            ):
                st.session_state.active_mood = mood_name
                st.rerun()

    if st.session_state.active_mood:
        mood_name = st.session_state.active_mood
        st.markdown(
            f'<div class="section-eyebrow">{len(MOODS)} moods available</div>'
            f'<div class="section-title">{MOODS[mood_name]["emoji"]} {mood_name} picks</div>',
            unsafe_allow_html=True,
        )

        mood_df = filter_by_mood(df, mood_name)
        mood_df = mood_df.sort_values("vote_average", ascending=False).head(20)
        mood_df = mood_df.rename(columns={
            "title_x": "Title", "vote_average": "Rating", "poster_url": "Poster"
        })
        mood_df["Genres"] = mood_df["genres_list"].apply(lambda x: ", ".join(x[:4]))
        mood_df["Overview"] = mood_df["overview"].fillna("").astype(str).str.slice(0, 250)

        render_movie_grid(mood_df, columns=5, show_similarity=False, key_prefix=f"mood_{mood_name}")
    else:
        st.caption("Pick a mood above to see matching movies.")


# ---------------------------------------------------------------------------
# Tab 3: surprise me
# ---------------------------------------------------------------------------
with tab_surprise:
    st.markdown(
        '<div class="section-eyebrow">Feeling indecisive?</div>'
        '<div class="section-title">Let the algorithm choose</div>',
        unsafe_allow_html=True,
    )

    surprise_mood = st.selectbox(
        "Surprise me within a mood (optional)",
        ["Any mood"] + list(MOODS.keys()),
    )

    if st.button("🎲 Surprise Me!", type="primary"):
        pool = df if surprise_mood == "Any mood" else filter_by_mood(df, surprise_mood)
        if pool.empty:
            st.warning("Couldn't find a movie for that mood — try 'Any mood'.")
        else:
            st.session_state.surprise_pick = pool.sample(1).iloc[0]

    pick = st.session_state.surprise_pick
    if pick is not None:
        c1, c2 = st.columns([1, 2])
        with c1:
            poster = pick["poster_url"] if pd.notna(pick.get("poster_url")) else PLACEHOLDER_POSTER
            st.image(poster, use_container_width=True)
        with c2:
            st.markdown(f"## {pick['title_x']}")
            st.write(f"⭐ **{pick['vote_average']:.1f}** · {', '.join(pick['genres_list'][:4])}")
            if pick["director"]:
                st.caption(f"Directed by {pick['director']}")
            st.write(pick["overview"])

            pick_row = {
                "id": pick.get("id"),
                "Title": pick.get("title_x"),
                "Rating": pick.get("vote_average"),
                "Genres": ", ".join(pick.get("genres_list", [])[:4]),
                "Overview": str(pick.get("overview", ""))[:250],
                "Poster": poster,
                "Director": pick.get("director", ""),
            }

            b1, b2 = st.columns(2)
            with b1:
                if st.button("🔁 Roll Again", use_container_width=True):
                    pool = df if surprise_mood == "Any mood" else filter_by_mood(df, surprise_mood)
                    st.session_state.surprise_pick = pool.sample(1).iloc[0]
                    st.rerun()
            with b2:
                pid = pick.get("id")
                if pd.notna(pid) and is_in_watchlist(pid):
                    if st.button("✓ In Watchlist", key="surprise_wl", use_container_width=True):
                        remove_from_watchlist(pid)
                        st.rerun()
                else:
                    if st.button("+ Watchlist", key="surprise_wl", use_container_width=True):
                        add_to_watchlist(pick_row)
                        st.rerun()


# ---------------------------------------------------------------------------
# Tab 4: watchlist
# ---------------------------------------------------------------------------
with tab_watchlist:
    _init_watchlist()
    watchlist = st.session_state.watchlist

    st.markdown(
        '<div class="section-eyebrow">Saved for later</div>'
        '<div class="section-title">Your watchlist</div>',
        unsafe_allow_html=True,
    )

    if not watchlist:
        st.info("Your watchlist is empty. Add movies from the other tabs with the **+ Watchlist** button.")
    else:
        col_count, col_clear = st.columns([4, 1])
        with col_count:
            st.caption(f"{len(watchlist)} movie{'s' if len(watchlist) != 1 else ''} saved")
        with col_clear:
            if st.button("Clear all", use_container_width=True):
                st.session_state.watchlist = {}
                st.rerun()

        watchlist_df = pd.DataFrame(watchlist.values())
        render_movie_grid(watchlist_df, columns=5, show_similarity=False, key_prefix="watchlist")


# ---------------------------------------------------------------------------
# Tab 5: dataset analytics
# ---------------------------------------------------------------------------
with tab_analytics:
    st.markdown(
        '<div class="section-eyebrow">By the numbers</div>'
        '<div class="section-title">Dataset analytics</div>',
        unsafe_allow_html=True,
    )

    @st.cache_data
    def genre_counts_chart(df):
        genres = [g for sub in df["genres_list"] for g in sub]
        return pd.Series(genres).value_counts().head(10)

    c1, c2 = st.columns(2)
    with c1:
        genre_counts = genre_counts_chart(df)
        fig = px.bar(
            x=genre_counts.index, y=genre_counts.values, title="Top Genres",
            color_discrete_sequence=["#e0334d"],
        )
        fig.update_layout(
            xaxis_title="Genre", yaxis_title="Count",
            plot_bgcolor="#0a0a0c", paper_bgcolor="#0a0a0c", font_color="#f5f5f5",
            font_family="Inter, sans-serif",
            title_font_family="Oswald, sans-serif",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(
            df, x="vote_average", nbins=20, title="Rating Distribution",
            color_discrete_sequence=["#f4b740"],
        )
        fig.update_layout(
            plot_bgcolor="#0a0a0c", paper_bgcolor="#0a0a0c", font_color="#f5f5f5",
            font_family="Inter, sans-serif",
            title_font_family="Oswald, sans-serif",
        )
        st.plotly_chart(fig, use_container_width=True)