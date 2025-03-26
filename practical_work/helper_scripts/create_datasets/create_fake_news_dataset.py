import requests
import json
import pandas as pd
import datetime
from bs4 import BeautifulSoup

BASE_URL = "https://www.snopes.com/fact-check/?pagenum={}"

#SNOPES
def scrape_snopes(num_pages=5):
    articles = []
    today_date = datetime.date.today().strftime("%Y-%m-%d")

    for page in range(1, num_pages + 1):
        url = BASE_URL.format(page)
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            script_tag = soup.find("script", type="application/ld+json")

            if script_tag:
                try:
                    data = json.loads(script_tag.string)
                    for item in data.get("mainEntity", {}).get("itemListElement", []):
                        title = item.get("name", "No Title")
                        articles.append(["", title, today_date])

                except json.JSONDecodeError as e:
                    print(f"JSON Parsing Error on Page {page}: {e}")

            print(f"Scraped page {page}")

        else:
            print(f"Failed to fetch page {page} (Status Code: {response.status_code})")
            break

    return articles


def scrape_article_text(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        content_div = soup.find("div", class_="sc-dWlLAA gIDoZn")
        if content_div:
            return content_div.text.strip()

    return "No Content Available"

# news_articles = scrape_snopes(num_pages=5)
#
# df = pd.DataFrame(news_articles, columns=["title", "text", "date"])
#
# df.to_csv("./fake_datasets/snopes_news.csv", index=False)
#
# for article in news_articles:
#     print(article)
#
# print("Saved to snopes_news.csv")


#BEFOREITSNEWS
# BASE_URL = "https://beforeitsnews.com/"
# headers = {"User-Agent": "Mozilla/5.0"}
# response = requests.get(BASE_URL, headers=headers)
#
# if response.status_code == 200:
#     print(response.text)
# else:
#     print(f"Failed to fetch page (Status Code: {response.status_code})")


import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime

base_url = "https://beforeitsnews.com"

response = requests.get(base_url)
soup = BeautifulSoup(response.text, "html.parser")

articles = soup.find_all("li", class_="item")

with open('fake_datasets/beforeitsnews.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["title", "text", "date"])

    for article in articles:
        title = article.find("div", class_="title").get_text(strip=True)
        link = article.find("a", href=True)['href']

        article_page = requests.get(base_url + link)
        article_soup = BeautifulSoup(article_page.text, "html.parser")

        article_text = article_soup.find("div", class_="article-content").get_text(strip=True) if article_soup.find(
            "div", class_="article-content") else "No content available"
        date = article_soup.find("time")["datetime"] if article_soup.find("time") else datetime.today().strftime(
            '%Y-%m-%d')

        writer.writerow([title, article_text, date])

more_articles_url = "https://beforeitsnews.com/all_items_more"
response_more = requests.get(more_articles_url)
soup_more = BeautifulSoup(response_more.text, "html.parser")
more_articles = soup_more.find_all("li", class_="item")

with open('fake_datasets/beforeitsnews.csv', 'a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    for article in more_articles:
        title = article.find("div", class_="title").get_text(strip=True)
        link = article.find("a", href=True)['href']

        article_page = requests.get(base_url + link)
        article_soup = BeautifulSoup(article_page.text, "html.parser")

        article_text = article_soup.find("div", class_="article-content").get_text(strip=True) if article_soup.find(
            "div", class_="article-content") else "No content available"
        date = article_soup.find("time")["datetime"] if article_soup.find("time") else datetime.today().strftime(
            '%Y-%m-%d')

        writer.writerow([title, article_text, date])

print("Scraping complete! Data saved to 'beforeitsnews.csv'.")

# UPDATE FAKE.CSV
# fake_df = pd.read_csv("./fake_datasets/Fake.csv").drop(columns=["subject"])
# snopes_df = pd.read_csv("./fake_datasets/snopes_news.csv")
#
# updated_df = pd.concat([fake_df, snopes_df], ignore_index=True)
# updated_df.to_csv("./fake_datasets/Fake.csv", index=False)