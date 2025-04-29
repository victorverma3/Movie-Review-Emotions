import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import sys
import torch
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from models.nn import EkmanEmotionClassifer
from utils import LABEL_TO_EMOTION_MAP

if __name__ == "__main__":

    # Loads movie review data
    movie_review_df = pd.read_csv("../data/processed/cleaned_movie_review_data.csv")

    # Configures embedding model
    embedding_model = SentenceTransformer(f"sentence-transformers/all-MiniLM-L6-v2")

    # Creates movie review embeddings
    directory = f"../data/output/embeddings"
    if not os.path.exists(directory):
        os.makedirs(directory)

    embeddings_path = "../data/output/embeddings/movie_review_embeddings.npy"
    if not os.path.exists(embeddings_path):
        embeddings = []
        for sentence in tqdm(
            movie_review_df["review_text"].tolist(),
            desc=f"Embedding movie reviews",
        ):
            embeddings.append(embedding_model.encode(sentence))
        embeddings = np.array(embeddings)

        embeddings_path = "../data/output/embeddings/movie_review_embeddings.npy"
        np.save(
            file=embeddings_path,
            arr=embeddings,
        )
        print(f"Created movie review embeddings at {embeddings_path}")

    # Loads movie review embeddings
    movie_review_embeddings = np.load(embeddings_path)
    print("Loaded movie review embeddings")

    # Loads classification model
    model = EkmanEmotionClassifer()
    model.load_state_dict(state_dict=torch.load("./best_nn_combined.pth"))

    # Classifies movie review emotions
    predictions = model.predict(movie_review_embeddings)
    predictions = [LABEL_TO_EMOTION_MAP[pred] for pred in list(predictions)]
    movie_review_df["predicted_emotion"] = predictions

    # Saves predictions
    save_path = "../data/output/predicted_movie_review_emotions.csv"
    movie_review_df.to_csv(save_path, index=False)
    print(f"Saved emotion predictions to {save_path}")
