import argparse
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import sys
from tqdm import tqdm
from typing import Sequence


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils import load_emotion_data_splits


# Gets bow embedding model
def get_bow_model(documents: Sequence[str]) -> None:

    if os.path.exists("./bow_model.pkl"):
        # Loads existing bow model
        with open("bow_model.pkl", "rb") as f:
            bow_model = pickle.load(f)
        print("Loaded bow embedding model")
    else:
        # Creates new bow model
        bow_model = CountVectorizer(max_features=1000)
        bow_model.fit(documents)

        with open("bow_model.pkl", "wb") as f:
            pickle.dump(bow_model, f)
        print("Created bow embedding model")

    return bow_model


# Creates embeddings
def create_embeddings(
    embedding_model: CountVectorizer | SentenceTransformer,
    embedding_model_name: str,
    data: Sequence[str],
    data_variant: str,
) -> None:

    directory = f"../data/neural_network/embeddings/{embedding_model_name}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Creates embeddings
    if embedding_model_name == "all-MiniLM-L6-v2":
        embeddings = []
        for sentence in tqdm(data, desc=f"Embedding {data_variant} data"):
            embeddings.append(embedding_model.encode(sentence))
        embeddings = np.array(embeddings)

    elif embedding_model_name == "bow":
        embeddings = embedding_model.transform(data).toarray()
        embeddings = np.array(embeddings)

    # Saves embeddings
    np.save(
        file=f"../data/neural_network/embeddings/{embedding_model_name}/{data_variant}_embeddings.npy",
        arr=embeddings,
    )
    print(f"Created {data_variant} embeddings")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Embedding model
    parser.add_argument(
        "-embed",
        "--embedding-model-name",
        help="The vector embedding model.",
        choices=["bow", "all-MiniLM-L6-v2"],
        default="all-MiniLM-L6-v2",
    )

    args = parser.parse_args()

    # Loads emotion data splits
    x_train, _, x_test, _, x_val, _ = load_emotion_data_splits()

    # Configures embedding model
    if args.embedding_model_name == "all-MiniLM-L6-v2":
        embedding_model = SentenceTransformer(
            f"sentence-transformers/{args.embedding_model_name}"
        )
    elif args.embedding_model_name == "bow":
        embedding_model = get_bow_model(documents=x_train)

    # Creates x_train embeddings
    create_embeddings(
        embedding_model=embedding_model,
        embedding_model_name=args.embedding_model_name,
        data=x_train,
        data_variant="x_train",
    )

    # Creates x_test embeddings
    create_embeddings(
        embedding_model=embedding_model,
        embedding_model_name=args.embedding_model_name,
        data=x_test,
        data_variant="x_test",
    )

    # Creates x_val embeddings
    create_embeddings(
        embedding_model=embedding_model,
        embedding_model_name=args.embedding_model_name,
        data=x_val,
        data_variant="x_val",
    )
