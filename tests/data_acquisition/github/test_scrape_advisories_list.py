import pytest
from pathlib import Path
import json
from unittest.mock import patch
from vdapa.data_acquisition.github.scrape_advisories_list import (
    load_existing_advisories,
    save_advisories_to_json,
    fetch_advisories_html,
    extract_last_page,
    parse_advisories_list,
    iso_to_datetime,
    scrape_all_advisories,
)

mock_dir = Path(__file__).parent / "mock_data"


def test_load_existing_advisories_file_not_exists(tmp_path):
    """Returns empty list if file does not exist."""
    file_path = tmp_path / "nonexistent.json"
    result = load_existing_advisories(file_path)
    assert result == []


def test_load_existing_advisories_file_exists(tmp_path):
    file_path = tmp_path / "existing.json"
    data = [{"ghsa": "GHSA-xxxx", "date": "2025-06-10T00:00:00Z"}]
    file_path.write_text(json.dumps(data))
    result = load_existing_advisories(file_path)
    assert result == data


def test_save_advisories_to_json(tmp_path):
    file_path = tmp_path / "output.json"
    data = [{"ghsa": "GHSA-xxxx"}]
    save_advisories_to_json(data, file_path)
    assert file_path.exists()
    content = json.loads(file_path.read_text())
    assert content == data


def test_fetch_advisories_html_invalid_page():
    """Raises ValueError if page is not positive integer."""
    with pytest.raises(ValueError):
        fetch_advisories_html(0)


@patch("vdapa.data_acquisition.github.scrape_advisories_list.requests.get")
def test_fetch_advisories_html_success(mock_get):
    mock_response = mock_get.return_value
    mock_response.raise_for_status.return_value = None
    mock_response.text = "<html></html>"
    html = fetch_advisories_html(1)
    assert html == "<html></html>"
    mock_get.assert_called_once()


def test_extract_last_page_none():
    assert extract_last_page("<html></html>") is None


def test_extract_last_page_valid():
    html = '<em data-total-pages="5"></em>'
    assert extract_last_page(html) == 5


def test_iso_to_datetime_valid():
    date_str = "2025-06-10T12:00:00Z"
    dt = iso_to_datetime(date_str)
    assert dt.year == 2025
    assert dt.month == 6
    assert dt.day == 10


def test_iso_to_datetime_invalid():
    assert iso_to_datetime("invalid-date") is None


def test_parse_advisories_list_type_error():
    with pytest.raises(TypeError):
        parse_advisories_list(123)


def test_parse_advisories_list_returns_list():
    html_path = mock_dir / "sample_advisories_page1.html"
    html = html_path.read_text()
    advisories = parse_advisories_list(html)
    assert isinstance(advisories, list)
    assert len(advisories) == 1
    assert advisories[0]['ghsa'] == "GHSA-xxxx"


def test_scrape_all_advisories_type_error():
    with pytest.raises(TypeError):
        scrape_all_advisories("not-a-date", delay=1)
    with pytest.raises(TypeError):
        scrape_all_advisories(None, delay="not-an-int")
