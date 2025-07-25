import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="vdapa CLI - Run data acquisition modules")
    parser.add_argument(
        "module",
        choices=["time_to_review_pipeline", "time_to_review_modeling", "time_to_review_report"],
        help="Module to run"
    )

    args = parser.parse_args()


    if args.module == "time_to_review_pipeline":
        from vdapa.data_acquisition.github.advisory_database_clone import run as clone_run
        from vdapa.preprocessing.normalize_advisory import run as normalize_run
        from vdapa.preprocessing.build_features_time_to_review import run as build_features_run
        from vdapa.modeling.time_to_review_estimator_bench import run as bench_run
        from vdapa.reporting.time_to_review_infer import run as infer_run
        from vdapa.reporting.time_to_review_report import run as report_run

        try:
            clone_run()
            normalize_run()
            build_features_run()
            bench_run()
            infer_run()
            report_run()
        except Exception as e:
            print(f"Error running time_to_review_pipeline: {e}", file=sys.stderr)
            sys.exit(1)

    if args.module == "time_to_review_modeling":
        from vdapa.preprocessing.build_features_time_to_review import run as build_features_time_to_review
        from vdapa.modeling.time_to_review_estimator_bench import run as bench_run
        from vdapa.reporting.time_to_review_infer import run as infer_run
        from vdapa.reporting.time_to_review_report import run as report_run

        try:
            build_features_time_to_review()
            bench_run()
            infer_run()
            report_run()
        except Exception as e:
            print(f"Error running time_to_review_model_report: {e}", file=sys.stderr)
            sys.exit(1)

    if args.module == "time_to_review_report":
        from vdapa.reporting.time_to_review_infer import run as infer_run
        from vdapa.reporting.time_to_review_report import run as report_run

        try:
            infer_run()
            report_run()
        except Exception as e:
            print(f"Error running time_to_review_report: {e}", file=sys.stderr)
            sys.exit(1)
if __name__ == "__main__":
    main()