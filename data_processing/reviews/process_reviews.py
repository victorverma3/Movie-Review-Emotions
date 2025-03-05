import emoji
from langdetect import detect, DetectorFactory
import pandas as pd
import re
import time
from tqdm import tqdm


month_map = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


# Cleans review text
def clean_review_text(row: pd.DataFrame) -> pd.DataFrame:

    # Creates additional date features
    day, month, year = row["date"].split(" ")
    row["month"] = month_map[month]
    row["day"] = int(day)
    row["year"] = int(year)

    # Converts emojis to text
    row["review_text"] = emoji.demojize(row["review_text"])

    # Removes non-English reviews
    try:
        DetectorFactory.seed = 0
        lang = detect(row["review_text"])
        if lang != "en":
            row["review_text"] = None

            return row
    except:
        pass

    # Removes extra whitespaces
    row["review_text"] = re.sub(r"\s+", " ", row["review_text"]).strip()

    return row


if __name__ == "__main__":

    start = time.perf_counter()

    # Loads movie review data
    movie_review_data = pd.read_csv("../../data/raw/all_movie_review_data.csv")

    # Drops reviews that were scraped as NaN
    movie_review_data.dropna(subset="review_text", inplace=True)

    # Cleans review text
    tqdm.pandas(desc="Cleaning review text")
    movie_review_data = movie_review_data.progress_apply(clean_review_text, axis=1)

    # Drops non-English reviews
    movie_review_data.dropna(subset="review_text", inplace=True)

    # Drops unnecessary features
    movie_review_data.drop(columns=["date"], inplace=True)

    # Saves processed movie review data
    movie_review_data.to_csv(
        "../../data/processed/cleaned_movie_review_data.csv", index=False
    )

    end = time.perf_counter()
    print(f"Processed {len(movie_review_data)} reviews in {end - start} seconds")
