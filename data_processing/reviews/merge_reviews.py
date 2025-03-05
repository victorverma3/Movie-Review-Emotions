import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":

    # Raw movie review data files
    base_path = "../../data/raw"
    files = [
        "movie_review_data_0_9.csv",
        "movie_review_data_10_19.csv",
        "movie_review_data_20_29.csv",
        "movie_review_data_30_39.csv",
        "movie_review_data_40_49.csv",
        "movie_review_data_50_59.csv",
        "movie_review_data_60_64.csv",
    ]

    # Concatenates all movie review data
    all_movie_reviews = pd.DataFrame()
    for i in tqdm(range(len(files)), desc="Concatenating movie review data"):
        movie_review_data = pd.read_csv(f"{base_path}/{files[i]}")
        all_movie_reviews = pd.concat([all_movie_reviews, movie_review_data], axis=0)

    # Saves concatenated movie review data
    all_movie_reviews.to_csv("../../data/raw/all_movie_review_data.csv", index=False)
