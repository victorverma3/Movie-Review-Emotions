import argparse
import numpy as np
import os
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils import load_emotion_data_embedings


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

    # Loads embeddings
    embedded_x_train, embedded_x_test, embedded_x_val = load_emotion_data_embedings(
        embedding_model_name=args.embedding_model
    )
