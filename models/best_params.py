import argparse
import json
import numpy as np
import os
import sys
from typing import Tuple


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


# Gets best neural network parameters from grid search results
def get_best_nn_parameters(
    embedding_model_name: str,
    training_dataset: str,
) -> Tuple[float, float, int, int, float]:

    directory = f"./nn_param_grid_search/{embedding_model_name}/{training_dataset}"
    grid_search_results = []

    # Concatenates grid search results
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                grid_search_results.append(json.load(f))

    # Retrieves best parameters
    best_lr = None
    best_momentum = None
    best_num_epochs = None
    best_batch_size = None
    best_f1 = 0

    for result in grid_search_results:
        mean_f1 = np.mean([result["test_f1_score"], result["val_f1_score"]])
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_lr = result["learning_rate"]
            best_momentum = result["momentum"]
            best_num_epochs = result["num_epochs"]
            best_batch_size = result["batch_size"]

    return best_lr, best_momentum, best_num_epochs, best_batch_size, best_f1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Training dataset
    parser.add_argument(
        "-td",
        "--training-dataset",
        help="Specifies the training dataset.",
        choices=["base", "combined"],
        default="combined",
    )

    # Embedding model name
    parser.add_argument(
        "-e",
        "--embedding_model_name",
        help="Specifies the training dataset.",
        choices=["all-MiniLM-L6-v2", "m3e-base"],
        default="all-MiniLM-L6-v2",
    )

    args = parser.parse_args()

    # Gets best neural network parameters
    learning_rate, momentum, num_epochs, batch_size, f1 = get_best_nn_parameters(
        embedding_model_name=args.embedding_model_name,
        training_dataset=args.training_dataset,
    )
    print("Embedding model name:", args.embedding_model_name)
    print("Training dataset:", args.training_dataset)
    print("Best learning rate:", learning_rate)
    print("Best momentum:", momentum)
    print("Best num epochs:", num_epochs)
    print("Best batch size:", batch_size)
    print("Best mean of test and validation f1 score:", f1)
