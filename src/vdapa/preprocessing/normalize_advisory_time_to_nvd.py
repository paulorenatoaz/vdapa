"""
Preprocesses raw data from the GitHub Advisory Database repository by
consolidating all advisory JSON files into a single tabular dataset with
normalized fields.

This module recursively scans the `reviewed` and `unreviewed` folders from the
cloned advisory-database, extracts relevant fields from each advisory, infers
its review status, and saves the result in JSON Lines format (and optionally CSV).

Designed to be called from the project entrypoint via the `run()` function.

"""
import re
from math import ceil
from pathlib import Path
import json

import numpy as np
from tqdm import tqdm
from dateutil.parser import parse as parse_date
from cvss import CVSS3, CVSS4

from vdapa.config import config, BASE_DIR
from vdapa.utils import setup_logging



logger = setup_logging("preprocessing", "preprocess_advisories")


def load_advisory_file(file_path):
    """Loads a single advisory JSON file.

    Args:
        file_path (Path): Path to the advisory JSON file.

    Returns:
        dict: Dictionary with the JSON content.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def extract_severity_scores(severity_list):
    """
    Given the `severity` field (a list of dicts), return a dict mapping each CVSS version
    to its numeric base score.

    Args:
        severity_list (list of dict): e.g.
            [
              {"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"},
              {"type": "CVSS_V4", "score": "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N"}
            ]

    Returns:
        dict: e.g. {"CVSS_V3": 7.8, "CVSS_V4": 9.8}
    """
    scores = {}
    for sev in severity_list or []:
        t = sev.get("type")
        raw = sev.get("score")
        if not (t and raw):
            continue
        try:
            if t.upper().startswith("CVSS_V3"):
                scores["CVSS_V3"] = CVSS3(raw).score()
            elif t.upper().startswith("CVSS_V4"):
                # CVSS4 scoring: use the .base_score attribute
                scores["CVSS_V4"] = CVSS4(raw).base_score
        except Exception:
            # parsing failed, skip
            continue
    return scores


def normalize_advisory(advisory_json):
    """Normalizes a single advisory JSON object.

    Args:
        advisory_json (dict): Raw advisory JSON data.
        status (str): Advisory status, 'reviewed' or 'unreviewed'.

    Returns:
        dict: Normalized advisory data with relevant fields.
    """
    published_at = advisory_json.get("published")
    db = advisory_json.get("database_specific", {})
    nvd_published_at = db.get("nvd_published_at")

    # parse dates
    published_at_parsed = parse_date(published_at) if published_at else None
    nvd_published_at_parsed = parse_date(nvd_published_at) if nvd_published_at else None

    # Calculate time to NVD
    delta = nvd_published_at_parsed - published_at_parsed if nvd_published_at else None
    time_to_nvd = ceil(delta.total_seconds() / 86400) if delta else None

    # Check if the advisory is preNVD
    if nvd_published_at:
        if time_to_nvd < 1:
            return None

    year = published_at_parsed.year if published_at_parsed else None
    month = published_at_parsed.month if published_at_parsed else None

    angle = 2 * np.pi * (month - 1) / 12

    month_sin = np.sin(angle)
    month_cos = np.cos(angle)

    github_reviewed_at = db.get("github_reviewed_at")
    github_reviewed_at_parsed = parse_date(github_reviewed_at) if github_reviewed_at else None

    delta = github_reviewed_at_parsed - published_at_parsed if github_reviewed_at_parsed else None
    time_to_review = ceil(delta.total_seconds() / 86400) if delta else None

    modified_at = advisory_json.get("modified")
    withdrawn_at = advisory_json.get("withdrawn")
    github_reviewed = db.get("github_reviewed")
    severity_label = db.get("severity")
    cwe_ids = db.get("cwe_ids", [])

    affected = advisory_json.get("affected", [])
    if affected and isinstance(affected, list):
        pkg = affected[0].get("package", {})
        ecosystem = pkg.get("ecosystem")
        package_name = pkg.get("name")
    else:
        ecosystem = None
        package_name = None

    aliases = advisory_json.get("aliases", [])

    # check if in aliases list there is a cve id

    cve_ids = []
    non_cve_aliases = []
    if aliases and isinstance(aliases, list):
        for alias in aliases:
            if alias.startswith("CVE-"):
                cve_ids.append(alias)
            else:
                non_cve_aliases.append(alias)

    for ref in advisory_json.get("references", []):
        url = ref.get("url", "")
        m = re.search(r"(CVE-\d{4}-\d{4,7})", url, re.IGNORECASE)
        if m:
            cve_ids.append(m.group(1).upper())

    cve_ids = list(set(cve_ids)) if cve_ids else None
    has_cve = True if cve_ids else False

    severity_list = advisory_json.get("severity", [])
    severity_score_list = extract_severity_scores(severity_list)
    severity_score_number = severity_score_list.get("CVSS_V3") or severity_score_list.get("CVSS_V4")
    has_withdrawn = withdrawn_at is not None

    return {
        "ghsa_id": advisory_json.get("id"),
        "published_at": published_at,
        "modified_at": modified_at,
        "github_reviewed": github_reviewed,
        "github_reviewed_at": github_reviewed_at,
        "cve_ids": cve_ids,
        "non_cve_aliases": non_cve_aliases,
        "has_cve": has_cve,
        "nvd_published_at": nvd_published_at,
        "time_to_review": time_to_review,
        "time_to_nvd": time_to_nvd,
        "has_withdrawn": has_withdrawn,
        "withdrawn_at": withdrawn_at,
        "severity_label": severity_label,
        "cvss_number": severity_score_number,
        "cwe_ids": cwe_ids,
        "num_cwes": len(cwe_ids) if isinstance(cwe_ids, list) else 0,
        "num_refs": len(advisory_json.get("references", [])),
        "year": year,
        "month": month,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "severity": advisory_json.get("severity"),
        "ecosystem": ecosystem,
        "package_name": package_name,
        "details": advisory_json.get("details"),
        "references": advisory_json.get("references", []),
        "affected_versions": advisory_json.get("affected", []),
        "summary": advisory_json.get("summary"),
    }


def load_all_advisories(input_folder):
    """Loads and normalizes all advisories from reviewed and unreviewed folders.

    Args:
        input_folder (Path): Root folder containing advisory JSONs.

    Returns:
        list: List of normalized advisory dictionaries.
    """
    advisories = []

    all_files = list(tqdm(input_folder.rglob("*.json"), desc="Listing advisory files"))

    for file_path in tqdm(all_files, desc="Reading and normalizing advisories"):
        try:
            advisory_json = load_advisory_file(file_path)
            normalized = normalize_advisory(advisory_json)
            if normalized:
                advisories.append(normalized)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    return advisories


def write_advisories_sorted_streaming(advisories, output_path, batch_size=1000):
    """Writes normalized advisories to file in JSON Lines format, sorted by published date.

    Args:
        advisories (list): List of normalized advisories.
        output_path (Path): Path to the final output file.
        batch_size (int): Number of advisories to write per batch.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    advisories_sorted = list(tqdm(
        sorted(advisories, key=lambda a: a["published_at"] or ""),
        desc="Sorting advisories by published_at"
    ))

    with open(output_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, len(advisories_sorted), batch_size), desc="Writing to JSON Lines"):
            batch = advisories_sorted[i:i + batch_size]
            for record in batch:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Advisories written to {output_path}")


def run():
    """Runs the advisories preprocessing pipeline."""
    logger.info("Starting advisory preprocessing...")

    input_folder = BASE_DIR / config["paths"]["raw_data"] / "advisory-database"
    output_folder = BASE_DIR / config["paths"]["processed_data"]
    output_file_json = output_folder / "advisories_normalized.json"
    # output_file_csv = output_folder / "advisories_normalized.csv"

    logger.info(f"Reading advisories from {input_folder}")
    advisories = load_all_advisories(input_folder)

    logger.info("Writing normalized dataset in streaming mode (JSON Lines)...")
    write_advisories_sorted_streaming(advisories, output_file_json)

    # logger.info("Also saving to CSV for exploration...")
    # df = pd.DataFrame(advisories)
    # df.sort_values(by="published_at", inplace=True)
    # df.to_csv(output_file_csv, index=False)

    logger.info("Advisory preprocessing completed successfully.")


if __name__ == "__main__":
    run()