import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from data_processing.scrape_movie_data import scrape_movie_data


# Plots emotion count by release decade
def plot_emotions_by_release_decade(df: pd.DataFrame) -> None:

    # Groups by decade
    df["release_decade"] = (df["release_year"] // 10) * 10
    emotion_counts = (
        df.groupby(["release_decade", "predicted_emotion"])
        .size()
        .reset_index(name="count")
    )
    pivot_df = emotion_counts.pivot(
        index="release_decade", columns="predicted_emotion", values="count"
    ).fillna(0)

    # Creates plot
    pivot_df.plot(kind="barh", stacked=True, figsize=(8, 6))
    plt.title("Emotion Counts by Release Decade")
    plt.xlabel("Review Count")
    plt.ylabel("Release Decade")
    plt.xticks(rotation=45)
    plt.legend(title="Predicted Emotion")
    plt.tight_layout()
    plt.savefig("./figures/emotion_counts_by_release_decade.png")


# plots average likes per emotion
def plot_average_likes_per_emotion(df: pd.DataFrame) -> None:

    # Calculates average likes
    avg_likes = (
        df.groupby("predicted_emotion")["num_likes"].mean().sort_values(ascending=False)
    )

    # Plot
    plt.figure(figsize=(8, 6))
    plt.bar(avg_likes.index, avg_likes.values)
    plt.xlabel("Emotion")
    plt.ylabel("Average Number of Likes")
    plt.title("Average Review Likes per Emotion")
    plt.tight_layout()
    plt.savefig("./figures/average_likes_per_emotion.png")


if __name__ == "__main__":

    # Loads movie review emotion predictions
    output_movie_df = pd.read_csv("../data/output/predicted_movie_review_emotions.csv")

    print(output_movie_df["predicted_emotion"].value_counts())

    # Gets movie data
    if os.path.exists("../data/processed/movie_data.csv"):
        movie_data_df = pd.read_csv("../data/processed/movie_data.csv")
    else:
        urls = output_movie_df["base_url"].unique()
        movie_data_df = scrape_movie_data(urls)

    # Combines data
    output_movie_df.rename(columns={"base_url": "url"}, inplace=True)
    movie_df = pd.merge(output_movie_df, movie_data_df, on="url")

    # Creates plots
    plot_emotions_by_release_decade(df=movie_df)
    plot_average_likes_per_emotion(df=movie_df)
