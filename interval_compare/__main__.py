import argparse
from datetime import date


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="interval_compare")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download")
    download.add_argument("--out", default="data")
    download.add_argument("--start", default="2021-01-01")
    download.add_argument("--end", default=None)

    run = commands.add_parser("run")
    run.add_argument("--snapshot", required=True)
    run.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.command == "download":
        from interval_compare.download import download_snapshot

        print(download_snapshot(args.out, start=args.start, end=args.end or date.today().isoformat()))
        return

    from interval_compare.forecast import load_forecaster
    from interval_compare.report import write_run

    write_run(args.snapshot, args.out, forecaster=load_forecaster())
    print(args.out)


if __name__ == "__main__":
    main()
