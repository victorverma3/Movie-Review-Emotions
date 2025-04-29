import pandas as pd

if __name__ == "__main__":

    # Loads emotion data
    labeled_emotions = pd.read_csv(
        "../../data/processed/cleaned_labeled_emotion_data.csv"
    )

    # Basic data statistics
    print(labeled_emotions.head())
    print("\nNumber of labeled emotion samples:", len(labeled_emotions))

    # Emotion distribution
    print(f'\n{labeled_emotions["emotion"].value_counts()}')

    # Loads emotion training data
    labeled_emotions_train = pd.read_csv(
        "../../data/processed/cleaned_labeled_emotion_data_train.csv"
    )

    # Basic data statistics
    print(f"\n{labeled_emotions_train.head()}")
    print("\nNumber of labeled emotion training samples:", len(labeled_emotions_train))

    # Emotion distribution
    print(f'\n{labeled_emotions_train["emotion"].value_counts()}')

    # Supplemental emotion data
    print("----------")
    # Loads emotion data
    supplemental_labeled_emotions = pd.read_csv(
        "../../data/processed/cleaned_labeled_emotion_data_supplemental.csv"
    )

    # Basic data statistics
    print(supplemental_labeled_emotions.head())
    print(
        "\nNumber of supplemental labeled emotion samples:",
        len(supplemental_labeled_emotions),
    )

    # Emotion distribution
    print(f'\n{supplemental_labeled_emotions["emotion"].value_counts()}')

    # Loads emotion training data
    supplemental_labeled_emotions_train = pd.read_csv(
        "../../data/processed/cleaned_labeled_emotion_data_train_supplemental.csv"
    )

    # Basic data statistics
    print(f"\n{supplemental_labeled_emotions_train.head()}")
    print(
        "\nNumber of supplemental labeled emotion training samples:",
        len(supplemental_labeled_emotions_train),
    )

    # Emotion distribution
    print(f'\n{supplemental_labeled_emotions_train["emotion"].value_counts()}')

    # Combined emotion data
    print("----------")
    # Loads emotion data
    combined_labeled_emotions = pd.concat(
        [labeled_emotions, supplemental_labeled_emotions], ignore_index=True
    )

    # Basic data statistics
    print(combined_labeled_emotions.head())
    print(
        "\nNumber of combined labeled emotion samples:",
        len(combined_labeled_emotions),
    )

    # Emotion distribution
    print(f'\n{combined_labeled_emotions["emotion"].value_counts()}')

    # Loads emotion training data
    combined_labeled_emotions_train = pd.concat(
        [
            pd.read_csv("../../data/processed/cleaned_labeled_emotion_data_train.csv"),
            pd.read_csv(
                "../../data/processed/cleaned_labeled_emotion_data_train_supplemental.csv"
            ),
        ],
        ignore_index=True,
    )

    # Basic data statistics
    print(f"\n{combined_labeled_emotions_train.head()}")
    print(
        "\nNumber of combined labeled emotion training samples:",
        len(combined_labeled_emotions_train),
    )

    # Emotion distribution
    print(f'\n{combined_labeled_emotions_train["emotion"].value_counts()}')
