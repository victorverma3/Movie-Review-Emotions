import argparse
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import sys
from tqdm import tqdm
from typing import Sequence


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils import load_emotion_data_splits


# Creates embeddings
def create_embeddings(
    embedding_model: str,
    embedding_model_name: str,
    data: Sequence,
    data_variant: str,
) -> None:

    # Creates embeddings
    embeddings = []
    for sentence in tqdm(data, desc=f"Embedding {data_variant} data"):
        embeddings.append(embedding_model.encode(sentence))
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
        "--embedding-model",
        help="The vector embedding model.",
        choices=["all-MiniLM-L6-v2"],
        default="all-MiniLM-L6-v2",
    )

    args = parser.parse_args()

    # Loads emotion data splits
    x_train, _, x_test, _, x_val, _ = load_emotion_data_splits()

    # Configures embedding model
    embedding_model = SentenceTransformer(
        f"sentence-transformers/{args.embedding_model}"
    )

    # Creates x_train embeddings
    create_embeddings(
        embedding_model=embedding_model,
        embedding_model_name=args.embedding_model,
        data=x_train,
        data_variant="x_train",
    )

    # Creates x_test embeddings
    create_embeddings(
        embedding_model=embedding_model,
        embedding_model_name=args.embedding_model,
        data=x_test,
        data_variant="x_test",
    )

    # Creates x_val embeddings
    create_embeddings(
        embedding_model=embedding_model,
        embedding_model_name=args.embedding_model,
        data=x_val,
        data_variant="x_val",
    )
