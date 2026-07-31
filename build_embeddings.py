import pandas as pd
import joblib
import ast

from sentence_transformers import SentenceTransformer

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

credits.rename(columns={"movie_id":"id"}, inplace=True)

df = movies.merge(credits, on="id")

def parse_names(x):
    try:
        return [i["name"] for i in ast.literal_eval(x)]
    except:
        return []

def director(x):
    try:
        crew = ast.literal_eval(x)
        for c in crew:
            if c["job"] == "Director":
                return c["name"]
    except:
        pass
    return ""

df["genres_list"] = df["genres"].apply(parse_names)
df["cast_list"] = df["cast"].apply(parse_names)
df["director"] = df["crew"].apply(director)

df["combined_text"] = (
    df["overview"].fillna("") + " " +
    df["genres_list"].apply(lambda x: " ".join(x)) + " " +
    df["cast_list"].apply(lambda x: " ".join(x[:5])) + " " +
    df["director"]
)

model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

embeddings = model.encode(
    df["combined_text"].tolist(),
    convert_to_numpy=True,
    batch_size=64,
    show_progress_bar=True
)

joblib.dump(embeddings, "embeddings.pkl")

print("embeddings.pkl created successfully")