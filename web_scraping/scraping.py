import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime as dt


def get_beautiful_soup_from_url(url: str) -> BeautifulSoup:
    html = requests.get(url)
    return BeautifulSoup(html.text, "html.parser")


def text_date_to_datetime(text: str) -> dt.datetime:
    if (
        "hours ago" in text.lower()
        or "hour ago" in text.lower()
        or "minutes ago" in text.lower()
        or "minute ago" in text.lower()
        or "just now" in text.lower()
    ):
        datetime = dt.datetime.now().date()
    elif "a day ago" in text.lower():
        datetime = dt.datetime.now().date() - dt.timedelta(days=1)
    elif "days ago" in text.lower():
        datetime = dt.datetime.now().date() - dt.timedelta(days=int(text[0]))
    else:
        if "Sept" in text:
            text = text.replace("t", "")
        datetime = dt.datetime.strptime(text, "%d %b %Y").date()

    return datetime


def get_trustpilot_reviews(soup: BeautifulSoup) -> pd.DataFrame:
    reviews = soup.find_all(
        class_="paper_paper__1PY90 paper_outline__lwsUX card_card__lQWDv card_noPadding__D8PcU styles_reviewCard__hcAvl"
    )

    review_titles = []
    review_texts = []
    reviewers = []
    review_ratings = []
    review_dates = []

    for review in reviews:
        review_title = review.find(
            class_="typography_heading-s__f7029 typography_appearance-default__AAY17"
        )
        review_text = review.find(
            class_="typography_body-l__KUYFJ typography_appearance-default__AAY17 typography_color-black__5LYEn"
        )

        reviewer_name = review.find(
            class_="typography_heading-xxs__QKBS8 typography_appearance-default__AAY17"
        )
        review_rating = review.find(
            class_="star-rating_starRating__4rrcf star-rating_medium__iN6Ty"
        ).findChild()

        review_date = (
            review.select_one(selector="time").getText().replace("Updated ", "")
        )
        review_date = text_date_to_datetime(text=review_date)
        review_dates.append(review_date)

        review_titles.append(review_title.getText())
        if review_text == None:
            review_texts.append("")
        else:
            review_texts.append(review_text.getText())
        reviewers.append(reviewer_name.getText())
        review_ratings.append(review_rating["alt"])

    df_reviews = pd.DataFrame(
        list(zip(reviewers, review_titles, review_dates, review_ratings, review_texts)),
        columns=[
            "reviewer",
            "review_title",
            "review_date",
            "review_rating",
            "review_text",
        ],
    )

    return df_reviews


def fetch_save_trustpilot_reviews(url: str, splitword: str | None) -> None:
    time = dt.datetime.now().strftime("%Y%m%d%H%M%S")

    soup = get_beautiful_soup_from_url(url=url)
    results_df = get_trustpilot_reviews(soup=soup)

    # Optionally get substring from url for adding page details to csv filename:
    if splitword:
        urltext = url.split(f"{splitword}")[1]
        results_df.to_csv(f"data/web_reviews/trustpilot_reviews_{urltext}_{time}.csv")

    else:
        results_df.to_csv(f"data/web_reviews/trustpilot_reviews_{time}.csv")


fetch_save_trustpilot_reviews(
    url="https://uk.trustpilot.com/review/octopus.energy?page=2", splitword="review/"
)
