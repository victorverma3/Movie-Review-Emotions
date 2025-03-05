import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
)
from transformers import pipeline
from tqdm import tqdm
from typing import Sequence

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


# Predicts emotion using baseline model
def predict_baseline_emotion(
    row: pd.DataFrame, baseline_model: pipeline
) -> pd.DataFrame:

    row["predicted_emotion"] = EMOTION_TO_LABEL_MAP.get(
        baseline_model(row["text"])[0]["label"], "neutral"
    )

    return row


# Evaluates the model
def evaluate_model(actual: Sequence[int], predicted: Sequence[int]):

    # Calculates common classification metrics
    metrics = classification_report(
        y_true=actual,
        y_pred=predicted,
        target_names=EKMAN_EMOTIONS,
        zero_division=0,
        output_dict=True,
    )

    with open(
        "../../data/baseline/baseline_emotion_classification_report.json", "w"
    ) as f:
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
    plt.title("Baseline Emotion Confusion Matrix")
    plt.savefig("../../data/baseline/figures/baseline_emotion_confusion_matrix.png")
    plt.close()


if __name__ == "__main__":

    # Loads labeled emotion data
    labeled_emotion_data = pd.read_csv(
        "../../data/processed/cleaned_labeled_emotion_data.csv"
    )

    # Performs baseline predictions
    baseline_model = pipeline(
        "sentiment-analysis", model="arpanghoshal/EkmanClassifier"
    )
    tqdm.pandas(desc="Predicting baseline emotions")
    predicted_emotion_data = labeled_emotion_data.progress_apply(
        predict_baseline_emotion, baseline_model=baseline_model, axis=1
    )

    # Drops neutral predictions
    predicted_emotion_data = predicted_emotion_data[
        ~(predicted_emotion_data["predicted_emotion"] == "neutral")
    ]

    # Saves baseline predictions
    predicted_emotion_data.to_csv(
        "../../data/baseline/baseline_emotion_predictions.csv", index=False
    )

    # Evaluates the model
    evaluate_model(
        actual=predicted_emotion_data["emotion"].tolist(),
        predicted=predicted_emotion_data["predicted_emotion"].tolist(),
    )
