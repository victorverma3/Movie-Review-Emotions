import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils import load_emotion_data_splits, evaluate_model

if __name__ == "__main__":

    # Loads emotion data splits
    x_train, y_train, x_test, y_test, x_val, y_val = load_emotion_data_splits()
    print("Loaded emotion data")

    # Trains Naive Bayes model using Term Frequency - Inverse Document Frequency
    tf_idf_nb_model = Pipeline([("tfidf", TfidfVectorizer()), ("nb", MultinomialNB())])
    tf_idf_nb_model.fit(x_train, y_train)
    print("Fit Naive Bayes model using TF-IDF")

    # TF-IDF test predictions
    y_pred = tf_idf_nb_model.predict(x_test)
    predicted_test_tf_idf_nb_emotion_data = pd.DataFrame(
        {"text": x_test, "emotion": y_test, "predicted_emotion": y_pred}
    )
    predicted_test_tf_idf_nb_emotion_data.to_csv(
        "../data/naive_bayes/tf_idf/test_emotion_predictions.csv", index=False
    )

    evaluate_model(
        actual=y_test,
        predicted=y_pred,
        classification_report_save_path="../data/naive_bayes/tf_idf/test_nb_emotion_classification_report.json",
        confusion_matrix_title="TF-IDF Naive Bayes Emotion Confusion Matrix (Test)",
        confusion_matrix_save_path="../data/naive_bayes/tf_idf/test_nb_emotion_confusion_matrix.png",
    )
    print("Evaluated TF-IDF NB test predictions")

    # TF-IDF validation predictions
    y_pred = tf_idf_nb_model.predict(x_val)
    predicted_val_tf_idf_nb_emotion_data = pd.DataFrame(
        {"text": x_val, "emotion": y_val, "predicted_emotion": y_pred}
    )
    predicted_test_tf_idf_nb_emotion_data.to_csv(
        "../data/naive_bayes/tf_idf/val_emotion_predictions.csv", index=False
    )

    evaluate_model(
        actual=y_val,
        predicted=y_pred,
        classification_report_save_path="../data/naive_bayes/tf_idf/val_nb_emotion_classification_report.json",
        confusion_matrix_title="TF-IDF Naive Bayes Emotion Confusion Matrix (Validation)",
        confusion_matrix_save_path="../data/naive_bayes/tf_idf/val_nb_emotion_confusion_matrix.png",
    )
    print("Evaluated TF-IDF NB test predictions")

    # Trains Naive Bayes model using Bag of Words
    bow_nb_model = Pipeline([("tfidf", CountVectorizer()), ("nb", MultinomialNB())])
    bow_nb_model.fit(x_train, y_train)
    print("Fit Naive Bayes model using BoW")

    # BoW test predictions
    y_pred = bow_nb_model.predict(x_test)
    predicted_test_bow_nb_emotion_data = pd.DataFrame(
        {"text": x_test, "emotion": y_test, "predicted_emotion": y_pred}
    )
    predicted_test_bow_nb_emotion_data.to_csv(
        "../data/naive_bayes/bow/test_emotion_predictions.csv", index=False
    )

    evaluate_model(
        actual=y_test,
        predicted=y_pred,
        classification_report_save_path="../data/naive_bayes/bow/test_nb_emotion_classification_report.json",
        confusion_matrix_title="BoW Naive Bayes Emotion Confusion Matrix (Test)",
        confusion_matrix_save_path="../data/naive_bayes/bow/test_nb_emotion_confusion_matrix.png",
    )
    print("Evaluated BoW NB test predictions")

    # BoW validation predictions
    y_pred = bow_nb_model.predict(x_val)
    predicted_val_bow_nb_emotion_data = pd.DataFrame(
        {"text": x_val, "emotion": y_val, "predicted_emotion": y_pred}
    )
    predicted_val_bow_nb_emotion_data.to_csv(
        "../data/naive_bayes/bow/val_emotion_predictions.csv", index=False
    )

    evaluate_model(
        actual=y_val,
        predicted=y_pred,
        classification_report_save_path="../data/naive_bayes/bow/val_nb_emotion_classification_report.json",
        confusion_matrix_title="BoW Naive Bayes Emotion Confusion Matrix (Validation)",
        confusion_matrix_save_path="../data/naive_bayes/bow/val_nb_emotion_confusion_matrix.png",
    )
    print("Evaluated Bow NB test predictions")
