import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------
# 1. Load Dataset
# --------------------------------

df = pd.read_csv("data/movies_data.csv")

print("Original shape:", df.shape)


# --------------------------------
# 2. Remove duplicates
# --------------------------------

df = df.drop_duplicates(subset=["Name"])

print("After removing duplicates:", df.shape)


# --------------------------------
# 3. Handle missing values
# --------------------------------

columns = [
    "Genre",
    "Director",
    "Actor 1",
    "Actor 2",
    "Actor 3"
]

for column in columns:
    df[column] = df[column].fillna("")


# --------------------------------
# 4. Create combined features
# --------------------------------

df["tags"] = (
    df["Genre"].astype(str) + " " +
    df["Director"].astype(str) + " " +
    df["Actor 1"].astype(str) + " " +
    df["Actor 2"].astype(str) + " " +
    df["Actor 3"].astype(str)
)


# --------------------------------
# 5. Convert text into numbers
# --------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

vectors = vectorizer.fit_transform(df["tags"])


# --------------------------------
# 6. Calculate similarity
# --------------------------------

similarity = cosine_similarity(vectors)


# --------------------------------
# 7. Create movie index
# --------------------------------

movie_indices = pd.Series(
    df.index,
    index=df["Name"]
).drop_duplicates()


# --------------------------------
# 8. Save model
# --------------------------------

model_data = {
    "movies": df,
    "vectors": vectors,
    "similarity": similarity,
    "movie_indices": movie_indices
}

with open("model/recommender.pkl", "wb") as file:
    pickle.dump(model_data, file)


print("Model created successfully!")
print("Movies:", len(df))