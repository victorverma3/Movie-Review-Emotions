import pandas as pd

if __name__ == "__main__":

    # Loads emotion data
    labeled_emotions = pd.read_csv(
        "../../data/processed/cleaned_labeled_emotion_data.csv"
    )
    print(f"Loaded {len(labeled_emotions)} labeled emotion observations")

    # Creates training data
    train_data = labeled_emotions.iloc[:13533]
    train_data.to_csv(
        "../../data/processed/cleaned_labeled_emotion_data_train.csv", index=False
    )
    print(f"Created {len(train_data)} training observations")

    # Creates test data
    test_data = labeled_emotions.iloc[13533:16433]
    test_data.to_csv(
        "../../data/processed/cleaned_labeled_emotion_data_test.csv", index=False
    )
    print(f"Created {len(test_data)} test observations")

    # Creates validation data
    validation_data = labeled_emotions.iloc[16433:]
    validation_data.to_csv(
        "../../data/processed/cleaned_labeled_emotion_data_validation.csv", index=False
    )
    print(f"Created {len(validation_data)} validation observations")
