import pandas as pd

if __name__ == "__main__":

    # Loads emotion data
    labeled_emotions = pd.read_csv(
        "../../data/processed/cleaned_labeled_emotion_data.csv"
    )

    # Basic data statistics
    print(f"\n{labeled_emotions.head()}")
    print(f"\n{labeled_emotions.describe()}")
    print("\nNumber of labeled emotion samples:", len(labeled_emotions))

    # Emotion distribution
    print(f'\n{labeled_emotions["emotion"].value_counts()}')
