import os
import pandas as pd
import sys
from transformers import pipeline
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from utils import (
    EMOTION_TO_LABEL_MAP,
    evaluate_model,
)


# Predicts emotion using baseline model
def predict_baseline_emotion(
    row: pd.DataFrame, baseline_model: pipeline
) -> pd.DataFrame:

    row["predicted_emotion"] = EMOTION_TO_LABEL_MAP.get(
        baseline_model(row["text"])[0]["label"], "neutral"
    )

    return row


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
        classification_report_save_path="../../data/baseline/baseline_emotion_classification_report.json",
        confusion_matrix_title="Baseline Emotion Confusion Matrix",
        confusion_matrix_save_path="../../data/baseline/baseline_emotion_confusion_matrix.png",
    )
