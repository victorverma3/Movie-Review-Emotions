import emoji
from langdetect import detect, DetectorFactory
import os
import pandas as pd
import re
import sys
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from utils import EMOJI_TO_EMOTION_MAP


# Cleans emotion text
def clean_emotion_text(row: pd.DataFrame) -> pd.DataFrame:

    # Converts emojis to text
    row["text"] = emoji.demojize(row["text"])

    # Removes non-English reviews
    try:
        DetectorFactory.seed = 0
        lang = detect(row["text"])
        if lang != "en":
            row["text"] = None

            return row
    except:
        pass

    # Removes extra whitespaces
    row["text"] = re.sub(r"\s+", " ", row["text"]).strip()

    return row


if __name__ == "__main__":

    # Parses labeled emotion data
    labeled_data = []
    with open("../../data/raw/twitter_emotion_corpus.txt") as f:
        for line in f:
            try:
                _, text, emotion = line.strip().split("\t")
                labeled_data.append(
                    {"text": text, "emotion": EMOJI_TO_EMOTION_MAP[emotion]}
                )
            except:
                continue
    print("Parsed labeled emotion data")

    # Converts emotion data to CSV
    labeled_emotion_data = pd.DataFrame(labeled_data)

    # Cleans emotion text
    tqdm.pandas(desc="Cleaning emotion data")
    labeled_emotion_data = labeled_emotion_data.progress_apply(
        clean_emotion_text, axis=1
    )

    # Drops non-English emotion text
    labeled_emotion_data.dropna(subset="text", inplace=True)

    # Saves processed labeled emotion data
    labeled_emotion_data.to_csv(
        "../../data/processed/cleaned_labeled_emotion_data_temp.csv", index=False
    )
    print("Saved labeled emotion data")
