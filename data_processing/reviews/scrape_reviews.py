import json
import pandas as pd
import requests as requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import time
from typing import Sequence
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager

ratings_map = {
    "½": 0.5,
    "★": 1,
    "★½": 1.5,
    "★★": 2,
    "★★½": 2.5,
    "★★★": 3,
    "★★★½": 3.5,
    "★★★★": 4,
    "★★★★½": 4.5,
    "★★★★★": 5,
}


# Loads base Letterboxd URLs
def load_base_movie_urls() -> Sequence[str]:

    with open("../../data/letterboxd_top_250_urls.json") as file:
        base_movie_urls = json.load(file)

    return base_movie_urls


# Configures up Selenium Chrome driver
def setup_driver() -> webdriver.Chrome:

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_extension("./ublockoriginlite.crx")

    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )


# Scrapes movie reviews
def scrape_movie_reviews(base_url: str, max_pages: int = 100) -> Sequence[dict]:

    movie_review_data = []
    driver = setup_driver()

    for page_number in tqdm(range(1, max_pages + 1), desc=base_url):
        review_url = f"{base_url}reviews/by/activity/page/{page_number}"
        driver.get(review_url)

        # Expands long reviews
        more_buttons = driver.find_elements(By.CSS_SELECTOR, "a.reveal")
        for button in more_buttons:
            try:
                button.click()
            except:
                continue

        # Scrapes all reviews
        reviews = driver.find_elements(By.CSS_SELECTOR, "div.film-detail-content")
        if not reviews:
            break

        for review in reviews:
            review_metadata = {}

            # Rating
            rating_elem = review.find_elements(By.CSS_SELECTOR, "span.rating")
            rating = rating_elem[0].text.strip() if rating_elem else -1.0

            # Date
            date_elem = review.find_elements(By.CSS_SELECTOR, "span._nobr")
            date = date_elem[0].text.strip() if date_elem else "N/A"

            # Number of comments
            comments_elem = review.find_elements(By.CSS_SELECTOR, "a.comment-count")
            num_comments = comments_elem[0].text.strip() if comments_elem else 0.0

            # Review text
            review_text_elem = review.find_elements(By.CSS_SELECTOR, "div.body-text")
            review_text = (
                review_text_elem[0].text.strip() if review_text_elem else "N/A"
            )

            # Number of likes
            likes_elem = review.find_elements(By.CSS_SELECTOR, "p.like-link-target")
            num_likes = (
                likes_elem[0].get_attribute("data-count") if likes_elem else -1.0
            )

            # Stores review data
            review_metadata["base_url"] = base_url
            review_metadata["rating"] = ratings_map.get(rating, -1.0)
            review_metadata["date"] = date
            review_metadata["num_comments"] = num_comments
            review_metadata["review_text"] = review_text
            review_metadata["num_likes"] = num_likes
            movie_review_data.append(review_metadata)

    return movie_review_data


if __name__ == "__main__":

    start = time.perf_counter()

    # Loads base movie urls
    base_movie_urls = load_base_movie_urls()

    # Scrapes movie review data
    start_url = 0
    end_url = 65
    all_movie_review_data = []
    for base_url in base_movie_urls[start_url:end_url]:
        movie_review_data = scrape_movie_reviews(base_url=base_url)
        all_movie_review_data.extend(movie_review_data)

    # Saves movie review data
    movie_review_data_df = pd.DataFrame(all_movie_review_data)
    movie_review_data_df.to_csv(
        f"../../data/raw/movie_review_data_{start_url}_{end_url-1}.csv", index=False
    )

    end = time.perf_counter()
    print(
        f"Scraped {len(all_movie_review_data)} reviews from {end_url - start_url} movie(s) in {end - start} seconds"
    )
