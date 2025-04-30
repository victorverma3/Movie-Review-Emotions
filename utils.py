import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from typing import Sequence, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EKMAN_EMOTIONS = ["Happiness", "Sadness", "Anger", "Fear", "Disgust", "Surprise"]

EMOTION_TO_LABEL_MAP = {
    "joy": 0,
    "sadness": 1,
    "anger": 2,
    "fear": 3,
    "disgust": 4,
    "surprise": 5,
}

LABEL_TO_EMOTION_MAP = {
    0: "happiness",
    1: "sadness",
    2: "anger",
    3: "fear",
    4: "disgust",
    5: "surprise",
}

EMOJI_TO_LABEL_MAP = {
    ":: joy": 0,
    ":: sadness": 1,
    ":: anger": 2,
    ":: fear": 3,
    ":: disgust": 4,
    ":: surprise": 5,
}

LABEL_TO_EMOJI_MAP = {
    0: ":: joy",
    1: ":: sadness",
    2: ":: anger",
    3: ":: fear",
    4: ":: disgust",
    5: ":: surprise",
}


# Loads train, test, and validation emotion data
def load_emotion_data_splits(
    training_dataset: str,
) -> Tuple[Sequence, Sequence, Sequence, Sequence, Sequence, Sequence]:

    if training_dataset == "base":
        # Loads train data
        train_data = pd.read_csv(
            os.path.join(
                BASE_DIR,
                "data/processed/cleaned_labeled_emotion_data_train.csv",
            )
        )
        x_train = train_data["text"].tolist()
        y_train = train_data["emotion"].tolist()

        # Loads test data
        test_data = pd.read_csv(
            os.path.join(
                BASE_DIR,
                "data/processed/cleaned_labeled_emotion_data_test.csv",
            )
        )
        x_test = test_data["text"].tolist()
        y_test = test_data["emotion"].tolist()

        # Loads validation data
        validation_data = pd.read_csv(
            os.path.join(
                BASE_DIR,
                "data/processed/cleaned_labeled_emotion_data_validation.csv",
            )
        )
        x_val = validation_data["text"].tolist()
        y_val = validation_data["emotion"].tolist()
    elif training_dataset == "supplemental":
        # Loads train data
        train_data = pd.read_csv(
            os.path.join(
                BASE_DIR,
                "data/processed/cleaned_labeled_emotion_data_train_supplemental.csv",
            )
        )
        x_train = train_data["text"].tolist()
        y_train = train_data["emotion"].tolist()

        # Loads test data
        test_data = pd.read_csv(
            os.path.join(
                BASE_DIR,
                "data/processed/cleaned_labeled_emotion_data_test_supplemental.csv",
            )
        )
        x_test = test_data["text"].tolist()
        y_test = test_data["emotion"].tolist()

        # Loads validation data
        validation_data = pd.read_csv(
            os.path.join(
                BASE_DIR,
                "data/processed/cleaned_labeled_emotion_data_validation_supplemental.csv",
            )
        )
        x_val = validation_data["text"].tolist()
        y_val = validation_data["emotion"].tolist()
    elif training_dataset == "combined":
        # Loads train data
        base_train_data = pd.read_csv(
            os.path.join(
                BASE_DIR,
                "data/processed/cleaned_labeled_emotion_data_train.csv",
            )
        )
        supplemental_train_data = pd.read_csv(
            os.path.join(
                BASE_DIR,
                "data/processed/cleaned_labeled_emotion_data_train_supplemental.csv",
            )
        )
        combined_train_data = pd.concat(
            [base_train_data, supplemental_train_data], ignore_index=True
        )
        x_train = combined_train_data["text"].tolist()
        y_train = combined_train_data["emotion"].tolist()

        # Loads test data
        base_test_data = pd.read_csv(
            os.path.join(
                BASE_DIR,
                "data/processed/cleaned_labeled_emotion_data_test.csv",
            )
        )
        supplemental_test_data = pd.read_csv(
            os.path.join(
                BASE_DIR,
                "data/processed/cleaned_labeled_emotion_data_test_supplemental.csv",
            )
        )
        combined_test_data = pd.concat(
            [base_test_data, supplemental_test_data], ignore_index=True
        )
        x_test = combined_test_data["text"].tolist()
        y_test = combined_test_data["emotion"].tolist()

        # Loads validation data
        base_validation_data = pd.read_csv(
            os.path.join(
                BASE_DIR,
                "data/processed/cleaned_labeled_emotion_data_validation.csv",
            )
        )
        supplemental_validation_data = pd.read_csv(
            os.path.join(
                BASE_DIR,
                "data/processed/cleaned_labeled_emotion_data_validation_supplemental.csv",
            )
        )
        combined_validation_data = pd.concat(
            [base_validation_data, supplemental_validation_data], ignore_index=True
        )
        x_val = combined_validation_data["text"].tolist()
        y_val = combined_validation_data["emotion"].tolist()

    print("Loaded emotion data")

    return x_train, y_train, x_test, y_test, x_val, y_val


# Loads emotion data embeddings
def load_emotion_data_embeddings(
    training_dataset: str,
    embedding_model_name: str,
) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:

    if training_dataset == "base":
        # Loads x_train embeddings
        embedded_x_train = np.load(
            os.path.join(
                BASE_DIR,
                f"data/nn/embeddings/{embedding_model_name}",
                "x_train_embeddings.npy",
            )
        )

        # Loads x_test embeddings
        embedded_x_test = np.load(
            os.path.join(
                BASE_DIR,
                f"data/nn/embeddings/{embedding_model_name}",
                "x_test_embeddings.npy",
            )
        )

        # Loads x_val embeddings
        embedded_x_val = np.load(
            os.path.join(
                BASE_DIR,
                f"data/nn/embeddings/{embedding_model_name}",
                "x_val_embeddings.npy",
            )
        )
    elif training_dataset == "combined":
        # Loads x_train embeddings
        embedded_x_train = np.load(
            os.path.join(
                BASE_DIR,
                f"data/nn/embeddings/{embedding_model_name}",
                "x_train_combined_embeddings.npy",
            )
        )

        # Loads x_test embeddings
        embedded_x_test = np.load(
            os.path.join(
                BASE_DIR,
                f"data/nn/embeddings/{embedding_model_name}",
                "x_test_combined_embeddings.npy",
            )
        )

        # Loads x_val embeddings
        embedded_x_val = np.load(
            os.path.join(
                BASE_DIR,
                f"data/nn/embeddings/{embedding_model_name}",
                "x_val_combined_embeddings.npy",
            )
        )

    print(f"Loaded {embedding_model_name} embeddings")

    return embedded_x_train, embedded_x_test, embedded_x_val


# Evaluates the model
def evaluate_model(
    actual: Sequence[int],
    predicted: Sequence[int],
    classification_report_save_path: str,
    confusion_matrix_title: str,
    confusion_matrix_save_path: str,
    save_metrics: bool = True,
    create_confusion_matrix: bool = True,
) -> Tuple[float, float, float, float]:

    # Calculates common classification metrics
    metrics = classification_report(
        y_true=actual,
        y_pred=predicted,
        target_names=EKMAN_EMOTIONS,
        zero_division=0,
        output_dict=True,
    )

    if save_metrics:
        with open(classification_report_save_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved classification report to {classification_report_save_path}")

    if create_confusion_matrix:
        # Computes the confusion matrix
        cm = confusion_matrix(
            y_true=actual, y_pred=predicted, labels=[0, 1, 2, 3, 4, 5]
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=EKMAN_EMOTIONS,
            yticklabels=EKMAN_EMOTIONS,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(confusion_matrix_title)
        plt.savefig(confusion_matrix_save_path)
        print(f"Saved confusion matrix to {confusion_matrix_save_path}")
        plt.close()

    return (
        metrics["accuracy"],
        metrics["macro avg"]["precision"],
        metrics["macro avg"]["recall"],
        metrics["macro avg"]["f1-score"],
    )


# Plots training losses
def plot_training_losses(losses: Sequence[float], save_path: str) -> None:

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(losses)), losses, marker="o", linestyle="-")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss over Time")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

    print(f"Saved training loss plot to {save_path}")
