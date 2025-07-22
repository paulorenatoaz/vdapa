import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="vdapa CLI - Run data acquisition modules")
    parser.add_argument(
        "module",
        choices=["scrape_advisories_list"],  # você pode adicionar outros módulos depois
        help="Module to run"
    )

    args = parser.parse_args()

    if args.module == "scrape_advisories_list":
        from vdapa.data_acquisition.github.scrape_advisories_list import run
        run()


if __name__ == "__main__":
    main()
