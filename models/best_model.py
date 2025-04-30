import argparse
import os
import sys
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from models.nn import EkmanEmotionClassifer
from utils import load_emotion_data_embeddings, load_emotion_data_splits

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

    # Save path
    parser.add_argument(
        "-sp",
        "--save_path",
        help="Defines the model save path.",
        default="./best_eec_model.pth",
    )

    args = parser.parse_args()

    # Loads classification model
    model = EkmanEmotionClassifer()
    model.load_state_dict(state_dict=torch.load(args.save_path))

    # Loads emotion data splits
    _, _, _, y_test, _, y_val = load_emotion_data_splits(
        training_dataset=args.training_dataset
    )

    # Loads embeddings
    _, embedded_x_test, embedded_x_val = load_emotion_data_embeddings(
        training_dataset=args.training_dataset,
        embedding_model_name=args.embedding_model_name,
    )

    # Evaluates test performance
    pred_test = model.predict(x=embedded_x_test)
    model.evaluate(
        true=y_test,
        pred=pred_test,
        data_variant="test",
        training_dataset=args.training_dataset,
        prediction_variant="best_predictions",
    )

    # Evaluates validation performance
    pred_val = model.predict(x=embedded_x_val)
    model.evaluate(
        true=y_val,
        pred=pred_val,
        data_variant="validation",
        training_dataset=args.training_dataset,
        prediction_variant="best_predictions",
    )
