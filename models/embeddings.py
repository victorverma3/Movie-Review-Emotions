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
    embedding_model: SentenceTransformer,
    data: Sequence[str],
    data_variant: str,
    training_dataset: str,
) -> None:

    directory = f"../data/nn/embeddings"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Creates embeddings
    embeddings = []
    for sentence in tqdm(data, desc=f"Embedding {data_variant} data"):
        embeddings.append(embedding_model.encode(sentence))
    embeddings = np.array(embeddings)

    # Saves embeddings
    if training_dataset == "base":
        np.save(
            file=f"../data/nn/embeddings/{data_variant}_embeddings.npy",
            arr=embeddings,
        )
    elif training_dataset == "combined":
        np.save(
            file=f"../data/nn/embeddings/{data_variant}_combined_embeddings.npy",
            arr=embeddings,
        )
    print(f"Created {data_variant} embeddings")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Training dataset
    parser.add_argument(
        "-td",
        "--training-dataset",
        help="Specifies the training dataset.",
        choices=["base", "combined"],
        default="base",
    )

    args = parser.parse_args()

    # Loads emotion data splits
    x_train, _, x_test, _, x_val, _ = load_emotion_data_splits(
        data_variant=args.training_dataset
    )

    # Configures embedding model
    embedding_model = SentenceTransformer(f"sentence-transformers/all-MiniLM-L6-v2")

    # Creates x_train embeddings
    create_embeddings(
        embedding_model=embedding_model,
        data=x_train,
        data_variant="x_train",
        training_dataset=args.training_dataset,
    )

    # Creates x_test embeddings
    create_embeddings(
        embedding_model=embedding_model,
        data=x_test,
        data_variant="x_test",
        training_dataset=args.training_dataset,
    )

    # Creates x_val embeddings
    create_embeddings(
        embedding_model=embedding_model,
        data=x_val,
        data_variant="x_val",
        training_dataset=args.training_dataset,
    )
