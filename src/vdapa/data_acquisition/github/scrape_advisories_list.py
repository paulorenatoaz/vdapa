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
    "User-Agent": config['github']['headers']['user_agent'],  # set a user agent in the config_private.yaml file
    "From": config['github']['headers']['from_email'],  # set an email in the config_private.yaml file
    "Referer": "https://github.com/advisories",
}

BASE_URL = config['github']['advisories_base_url']


def load_existing_advisories(filepath=RAW_DATA_PATH):
    """
    Load existing advisories from a JSON file.

    Args:
        filepath (Path or str): Path to the JSON file containing advisories.

    Returns:
        list: List of advisories loaded from the JSON file, or empty list if file does not exist.

    Raises:
        TypeError: If filepath is not str or Path.
        IOError: If the file exists but cannot be read.
        json.JSONDecodeError: If the file content is invalid JSON.
    """
    if not isinstance(filepath, (str, Path)):
        raise TypeError(f"Expected str or Path for filepath, got {type(filepath)}")

    filepath = Path(filepath)
    if not filepath.exists():
        return []

    try:
        with filepath.open('r', encoding='utf-8') as f:
            return json.load(f)
    except IOError as e:
        logger.error(f"Error reading file {filepath}: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filepath}: {e}")
        raise


def save_advisories_to_json(advisories, filepath=RAW_DATA_PATH):
    """
    Save advisories to a JSON file.

    Args:
        advisories (list): List of advisory dicts to save.
        filepath (Path or str): Path where the JSON will be saved.

    Raises:
        TypeError: If filepath is not str or Path.
        IOError: If unable to write to the file.
    """
    if not isinstance(filepath, (str, Path)):
        raise TypeError(f"Expected str or Path for filepath, got {type(filepath)}")

    filepath = Path(filepath)
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open('w', encoding='utf-8') as f:
            json.dump(advisories, f, indent=2)
        logger.info(f"Saved {len(advisories)} advisories to {filepath}")
    except IOError as e:
        logger.error(f"Error writing to file {filepath}: {e}")
        raise


def fetch_advisories_html(page):
    """
    Fetch the HTML content of a GitHub advisories page.

    Args:
        page (int): Page number to fetch. Must be positive integer.

    Returns:
        str: HTML content of the requested page.

    Raises:
        ValueError: If page is not a positive integer.
        requests.RequestException: If the HTTP request fails.
    """
    if not isinstance(page, int) or page < 1:
        raise ValueError(f"Page must be a positive integer, got {page}")

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
    """
    Extract the last page number from advisories HTML pagination.

    Args:
        html (str): HTML content of advisories page.

    Returns:
        int or None: Last page number if found, else None.

    Raises:
        TypeError: If html is not a string.
    """
    if not isinstance(html, str):
        raise TypeError(f"Expected str for html, got {type(html)}")

    soup = BeautifulSoup(html, "html.parser")
    em = soup.find("em", attrs={"data-total-pages": True})
    if em:
        try:
            last_page = int(em['data-total-pages'])
            logger.info(f"Last page found: {last_page}")
            return last_page
        except ValueError:
            logger.error("Invalid last page number found in HTML.")
            return None
    return None


def parse_advisories_list(html, newest_date=None):
    """
    Parse advisories list from HTML content.

    Args:
        html (str): HTML content of advisories page.
        newest_date (datetime.datetime or None): Advisories older than this date will be skipped.

    Returns:
        list: List of advisory dicts.

    Raises:
        TypeError: If html is not a string or newest_date is not None or datetime.
    """
    if not isinstance(html, str):
        raise TypeError(f"Expected str for html, got {type(html)}")
    if newest_date is not None and not isinstance(newest_date, datetime):
        raise TypeError(f"Expected datetime or None for newest_date, got {type(newest_date)}")

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
        if newest_date and date and iso_to_datetime(date) <= newest_date:
            logger.info(f"Found advisory older than newest date in base ({date} <= {newest_date}), GHSA: {ghsa}, stopping scrape current page.")
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
    """
    Convert ISO 8601 date string to datetime object.

    Args:
        date_str (str): Date string in ISO 8601 format.

    Returns:
        datetime.datetime or None: Parsed datetime object or None if parsing fails.

    Raises:
        TypeError: If date_str is not a string.
    """
    if not isinstance(date_str, str):
        raise TypeError(f"Expected str for date_str, got {type(date_str)}")

    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except Exception:
        return None


def scrape_all_advisories(newest_date, delay=int(config['scraping']['delay_seconds'])):
    """
    Scrape all advisories from GitHub, starting from the newest date.

    If newest_date is None, scrapes from the oldest advisories.

    Args:
        newest_date (datetime.datetime or None): Advisories older than this date are skipped.
        delay (int): Delay in seconds between page requests.

    Returns:
        list: List of scraped advisories.

    Raises:
        TypeError: If newest_date is not None or datetime, or delay is not int.
    """
    if newest_date is not None and not isinstance(newest_date, datetime):
        raise TypeError(f"Expected datetime or None for newest_date, got {type(newest_date)}")
    if not isinstance(delay, int):
        raise TypeError(f"Expected int for delay, got {type(delay)}")

    scraped_advisories = []

    if not newest_date:
        logger.info("No existing advisories found, starting fresh scrape.")
        last_page = extract_last_page(fetch_advisories_html(1))
        if not last_page:
            logger.error("Could not determine last page number, aborting.")
            return scraped_advisories

        page = last_page

        while page > 0:
            try:
                html = fetch_advisories_html(page)
                scraped_advisories_in_page = parse_advisories_list(html, newest_date)
                scraped_advisories_in_page.reverse()
                scraped_advisories.extend(scraped_advisories_in_page)

                page -= 1
                time.sleep(delay)

            except Exception as e:
                logger.error(f"Error processing page {page}: {e}")
                break

    else:
        logger.info(f"Found existing advisories, starting scrape from newest date: {newest_date}")
        page = 1

        while True:
            try:
                html = fetch_advisories_html(page)
                scraped_advisories_in_page = parse_advisories_list(html, newest_date)

                if not scraped_advisories_in_page:
                    logger.info(f"No new advisories found on page {page}, stopping.")
                    return scraped_advisories

                scraped_advisories_in_page.reverse()
                scraped_advisories[:0] = scraped_advisories_in_page

                page += 1
                time.sleep(delay)

            except Exception as e:
                logger.error(f"Error processing page {page}: {e}")
                break

    return scraped_advisories


def run():
    """
    Main entry point to scrape advisories and save them to JSON file.
    """
    existing_advisories = load_existing_advisories()
    if existing_advisories:
        newest_date = iso_to_datetime(existing_advisories[-1]['date'])
    else:
        newest_date = None

    scraped_advisories = scrape_all_advisories(newest_date)
    existing_advisories.extend(scraped_advisories)
    save_advisories_to_json(existing_advisories)


if __name__ == "__main__":
    run()
