# Imports
from bs4 import BeautifulSoup
import json
import os
import pandas as pd
import re
from requests_html import HTMLSession
import sys
from tqdm import tqdm
from typing import Sequence, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


# Scrapes movie data
def scrape_movie_data(movie_urls: Sequence[str]) -> pd.DataFrame:

    movie_data = []
    for url in tqdm(movie_urls, desc="Scraping movie data"):
        result = get_letterboxd_data(url=url)
        if result:
            movie_data.append(result)

    # Processes movie data
    movie_data_df = pd.DataFrame(movie_data)
    movie_data_df["genres"] = movie_data_df["genres"].apply(
        lambda genres: [genre.lower().replace(" ", "_") for genre in genres]
    )

    # Saves movie data
    movie_data_df.to_csv("../data/processed/movie_data.csv", index=False)

    return movie_data_df


# Gets Letterboxd data
def get_letterboxd_data(
    url: str,
) -> Tuple[str, str, int, int, float, int, int, int, str]:

    # Loads page
    session = HTMLSession()
    response = session.get(url, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Trims useful data
    try:
        script_tag = str(soup.find("script", {"type": "application/ld+json"}))
        script = script_tag[52:-20]
        webData = json.loads(script)
    except:

        return None

    # Scrapes relevant Letterboxd data
    try:
        title = webData["name"]  # Title
        release_year = int(webData["releasedEvent"][0]["startDate"])  # Release year
        runtime = int(
            re.search(
                r"(\d+)\s+mins", soup.find("p", {"class": "text-footer"}).text
            ).group(1)
        )  # Runtime
        rating = webData["aggregateRating"]["ratingValue"]  # Letterboxd rating
        rating_count = webData["aggregateRating"][
            "ratingCount"
        ]  # Letterboxd rating count
        genre = webData["genre"]  # Genres
        country = webData["countryOfOrigin"][0]["name"]  # country of origin
        poster = webData["image"]  # Poster
    except:

        return None

    return {
        "url": url,
        "title": title,
        "release_year": release_year,
        "runtime": runtime,
        "letterboxd_rating": rating,
        "letterboxd_rating_count": rating_count,
        "genres": genre,
        "country_of_origin": country,
        "poster": poster,
    }
