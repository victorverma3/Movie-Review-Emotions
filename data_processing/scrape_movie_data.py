# Imports
from bs4 import BeautifulSoup
import json
import os
import pandas as pd
import re
from requests_html import HTMLSession
import sys
import time
from tqdm import tqdm
from typing import Sequence, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


# Encodes genres as integers
def encode_genres(genres):

    genre_options = [
        "action",
        "adventure",
        "animation",
        "comedy",
        "crime",
        "documentary",
        "drama",
        "family",
        "fantasy",
        "history",
        "horror",
        "music",
        "mystery",
        "romance",
        "science_fiction",
        "tv_movie",
        "thriller",
        "war",
        "western",
    ]

    genre_binary = ""
    for genre in genre_options:
        if genre in genres:
            genre_binary += "1"
        else:
            genre_binary += "0"

    genre_int = int(genre_binary, 2)

    return genre_int


# Maps countries to numerical values
def assign_countries(country_of_origin):

    country_map = {
        "USA": 0,
        "UK": 1,
        "China": 2,
        "France": 3,
        "Japan": 4,
        "Germany": 5,
        "South Korea": 6,
        "Canada": 7,
        "India": 8,
        "Austrailia": 9,
        "Hong Kong": 10,
        "Italy": 11,
        "Spain": 12,
        "Brazil": 13,
        "USSR": 14,
    }

    return country_map.get(country_of_origin, len(country_map))


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
    movie_data_df["genres"] = movie_data_df["genres"].apply(encode_genres)
    movie_data_df["country_of_origin"] = movie_data_df["country_of_origin"].apply(
        assign_countries
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
