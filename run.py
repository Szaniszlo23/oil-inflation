"""
run.py - the only entry point.

  python run.py               run everything on the latest raw snapshot
                              (downloads one first if none exists)
  python run.py --refresh     download a new snapshot, then run everything
  python run.py --from build  rerun from a stage onwards

Stages that aren't written yet are skipped with a message.
"""
import argparse
import importlib
import sys

from pipeline import utils

STAGES = ["fetch", "build", "stage1", "stage2", "chain", "policy", "charts"]


def main():
    parser = argparse.ArgumentParser(description="Oil price pass-through pipeline")
    parser.add_argument("--refresh", action="store_true", help="download a new raw snapshot")
    parser.add_argument("--from", dest="start", choices=STAGES, default="fetch",
                        help="first stage to run")
    args = parser.parse_args()

    utils.ensure_dirs()
    cfg = utils.load_config()

    for stage in STAGES[STAGES.index(args.start):]:
        if stage == "fetch" and not args.refresh and utils.latest_snapshot():
            print(f"[fetch] using snapshot {utils.latest_snapshot().name} "
                  f"(run with --refresh to download a new one)")
            continue

        module = importlib.import_module(f"pipeline.{stage}")
        if not hasattr(module, "run"):
            print(f"[{stage}] not built yet - skipped")
            continue

        print(f"[{stage}]")
        try:
            module.run(cfg)
        except Exception as e:
            print(f"[{stage}] stopped: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()