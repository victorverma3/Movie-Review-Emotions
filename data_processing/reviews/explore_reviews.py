import pandas as pd

if __name__ == "__main__":

    # Loads movie review data
    movie_reviews = pd.read_csv("../../data/processed/cleaned_movie_review_data.csv")

    # Basic data statistics
    print(f"\n{movie_reviews.head()}")
    print(f"\n{movie_reviews.describe()}")
    print("\nNumber of movie reviews:", len(movie_reviews))

    # Rating distribution
    print(f'\n{movie_reviews["rating"].value_counts()}')
