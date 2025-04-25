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

EMOJI_TO_EMOTION_MAP = {
    ":: joy": 0,
    ":: sadness": 1,
    ":: anger": 2,
    ":: fear": 3,
    ":: disgust": 4,
    ":: surprise": 5,
}

EMOTION_TO_EMOJI_MAP = {
    0: ":: joy",
    1: ":: sadness",
    2: ":: anger",
    3: ":: fear",
    4: ":: disgust",
    5: ":: surprise",
}


# Loads train, test, and validation emotion data
def load_emotion_data_splits() -> (
    Tuple[Sequence, Sequence, Sequence, Sequence, Sequence, Sequence]
):

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

    print("Loaded emotion data")

    return x_train, y_train, x_test, y_test, x_val, y_val


# Loads emotion data embeddings
def load_emotion_data_embeddings(
    embedding_model_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Loads x_train embeddings
    embedded_x_train = np.load(
        os.path.join(
            BASE_DIR,
            "data/nn/embeddings",
            embedding_model_name,
            "x_train_embeddings.npy",
        )
    )

    # Loads x_test embeddings
    embedded_x_test = np.load(
        os.path.join(
            BASE_DIR,
            "data/nn/embeddings",
            embedding_model_name,
            "x_test_embeddings.npy",
        )
    )

    # Loads x_val embeddings
    embedded_x_val = np.load(
        os.path.join(
            BASE_DIR,
            "data/nn/embeddings",
            embedding_model_name,
            "x_val_embeddings.npy",
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
):

    # Calculates common classification metrics
    metrics = classification_report(
        y_true=actual,
        y_pred=predicted,
        target_names=EKMAN_EMOTIONS,
        zero_division=0,
        output_dict=True,
    )

    with open(classification_report_save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Computes the confusion matrix
    cm = confusion_matrix(y_true=actual, y_pred=predicted, labels=[0, 1, 2, 3, 4, 5])

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
    plt.close()
