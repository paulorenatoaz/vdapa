import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from vdapa.config import config, BASE_DIR
from vdapa.utils import setup_logging
import json
from datetime import datetime

logger = setup_logging("data_acquisition", "advisories_list")

RAW_DATA_PATH = BASE_DIR / Path(config['paths']['raw_data']) / 'github_advisories_list.json'

HEADERS = {
    "User-Agent": config['github']['headers']['user_agent'], # set a user agent in the config_private.yaml file
    "From": config['github']['headers']['from_email'], # set an email in the config_private.yaml file
    "Referer": "https://github.com/advisories",
}

BASE_URL = config['github']['advisories_base_url']

def load_existing_advisories(filepath=RAW_DATA_PATH):
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_advisories_to_json(advisories, filepath=RAW_DATA_PATH):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(advisories, f, indent=2)
    logger.info(f"Saved {len(advisories)} advisories to {filepath}")


def fetch_advisories_html(page):
    url = f"{BASE_URL}{page}"
    token = config['github'].get('token')
    headers = HEADERS.copy()
    if token:
        headers['Authorization'] = f"token {token}"
    try:
        logger.info(f"Fetching advisories page {page}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error(f"Failed to fetch page {page}: {e}")
        raise


def extract_last_page(html):
    soup = BeautifulSoup(html, "html.parser")
    em = soup.find("em", attrs={"data-total-pages": True})
    if em:
        try:
            last_page = int(em['data-total-pages'])
            logger.info(f"Last page found: {last_page}")
            return last_page
        except ValueError:
            return None
    return None


def parse_advisories_list(html, newest_date=None):
    soup = BeautifulSoup(html, "html.parser")
    advisories = []

    advisory_divs = soup.find_all("div", class_="Box-row Box-row--focus-gray p-0 js-navigation-item")

    for div in advisory_divs:
        title_tag = div.find("a", class_="Link--primary")
        if not title_tag:
            continue
        title = title_tag.text.strip()
        link = "https://github.com" + title_tag['href']
        ghsa = title_tag['href'].split("/advisories/")[-1] if "/advisories/" in title_tag['href'] else None

        date_tag = div.find("relative-time")
        date = date_tag['datetime'] if date_tag else None
        # if date is older than the newest_date, stop scraping
        if newest_date and date and iso_to_datetime(date) <= newest_date:
            logger.info(f"Found advisory  older than newest date in base ({date} <= {newest_date}), GHSA: {ghsa}, stopping scrape.")
            return advisories



        severity_tag = div.find("span", class_="Label")
        severity = severity_tag.text.strip() if severity_tag else None

        cve_tag = div.find("span", class_="text-bold")
        cve = cve_tag.text.strip() if cve_tag and cve_tag.text.strip().startswith("CVE-") else None

        package_tag = div.find_all("span")[-1] if div.find_all("span") else None
        package = package_tag.text.strip() if package_tag else None



        advisories.append({
            "ghsa": ghsa,
            "cve": cve,
            "title": title,
            "severity": severity,
            "package": package,
            "date": date,
            "link": link,
        })
    return advisories


def iso_to_datetime(date_str):
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except Exception:
        return None


def scrape_all_advisories(newest_date, delay=int(config['scraping']['delay_seconds'])):
    scraped_advisories = []
    if not newest_date:
        # No existing advisories, start fresh. Scraping from the last page, upwards, i.e., oldest first.
        logger.info("No existing advisories found, starting fresh scrape.")
        last_page = extract_last_page(fetch_advisories_html(0))
        if not last_page:
            logger.error("Could not determine last page number, aborting.")
            return scraped_advisories

        page = last_page

        while page > 0:
            try:
                html = fetch_advisories_html(page)
                scraped_advisories_in_page = parse_advisories_list(html, newest_date)
                scraped_advisories_in_page.reverse() # revert the order to start from the oldest advisories
                scraped_advisories.extend(scraped_advisories_in_page)

                page -= 1
                time.sleep(delay)

            except Exception as e:
                logger.error(f"Error processing page {page}: {e}")
                break

    else:
        # Existing advisories found. Scrape from the newest until the newest date.
        logger.info(f"Found existing advisories, starting scrape from newest date: {newest_date}")
        page = 1
        while True:
            try:
                html = fetch_advisories_html(page)
                scraped_advisories_in_page = parse_advisories_list(html, newest_date)

                if not scraped_advisories_in_page:
                    logger.info(f"No new advisories found on page {page}, stopping.")
                    return scraped_advisories

                scraped_advisories_in_page.reverse() # Reverse to start from the oldest advisories
                scraped_advisories[:0] = scraped_advisories_in_page  # Insert at the beginning to maintain order, oldest first

                page += 1
                time.sleep(delay)

            except Exception as e:
                logger.error(f"Error processing page {page}: {e}")
                break

    return scraped_advisories


def run():
    existing_advisories = load_existing_advisories()
    # get the newest date from existing advisories, i.e., the last entry in the list.
    if existing_advisories:
        newest_date = iso_to_datetime(existing_advisories[-1]['date'])
    else:
        newest_date = None

    scraped_advisories = scrape_all_advisories(newest_date)
    existing_advisories.extend(scraped_advisories)
    save_advisories_to_json(existing_advisories)


if __name__ == "__main__":
    run()
