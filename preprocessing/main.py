#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Scanpy/Numba otherwise tries to cache compiled functions beside the shared
# Conda installation, which is read-only on the compute nodes.  Set local
# defaults before importing the pipeline (and therefore Scanpy).
_RUNTIME_CACHE = Path(__file__).resolve().parent / ".cache" / "runtime"
for _name, _path in {
    "NUMBA_CACHE_DIR": _RUNTIME_CACHE / "numba",
    "MPLCONFIGDIR": _RUNTIME_CACHE / "matplotlib",
}.items():
    os.environ.setdefault(_name, str(_path))
    Path(os.environ[_name]).mkdir(parents=True, exist_ok=True)

from utils.config import ALL_LOGICAL_DATASETS, DATASET_GROUPS
from utils.pipeline import ReconstructionPipeline


def parse_dataset_selection(value: str) -> list[str]:
    """Resolve --only into an ordered list of logical datasets."""
    if not value or value.lower() == "all":
        return list(ALL_LOGICAL_DATASETS)

    selected: list[str] = []
    for token in (part.strip() for part in value.split(",")):
        if not token:
            continue
        selected.extend(DATASET_GROUPS.get(token.lower(), [token]))

    unknown = [name for name in selected if name not in ALL_LOGICAL_DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}")

    return list(dict.fromkeys(selected))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the scLightGAT Hugging Face dataset from public/original "
            "sources."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    build = subparsers.add_parser(
        "build", help="download, reconstruct, and write h5ad files"
    )
    build.add_argument("--cache-dir", default="./.cache")
    build.add_argument("--output", default="./scLightGAT_data")
    build.add_argument(
        "--only",
        default="all",
        help="all|main|train|test|extra|caf or comma-separated dataset names",
    )
    build.add_argument(
        "--keep-raw",
        action="store_true",
        help="keep downloaded source files after a successful build",
    )
    build.add_argument(
        "--force",
        action="store_true",
        help="discard checkpoints and rebuild the selected datasets",
    )
    build.add_argument(
        "--from-stage",
        choices=["parsed", "qc", "annotated", "normalized", "written"],
        default=None,
        help="invalidate this stage and every later stage before rebuilding",
    )
    build.add_argument("--seed", type=int, default=0)

    status = subparsers.add_parser("status", help="show checkpoint status")
    status.add_argument("--cache-dir", default="./.cache")
    status.add_argument("--output", default="./scLightGAT_data")

    clean = subparsers.add_parser(
        "clean", help="remove source/intermediate caches, never final outputs"
    )
    clean.add_argument("--cache-dir", default="./.cache")
    clean.add_argument("--output", default="./scLightGAT_data")
    clean.add_argument(
        "--intermediates",
        action="store_true",
        help="also remove computational intermediate h5ad files",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Running `python main.py` is equivalent to `python main.py build`.
    if args.command is None:
        args = parser.parse_args(["build"])

    pipeline = ReconstructionPipeline(
        cache_dir=Path(args.cache_dir),
        output=Path(args.output),
    )

    if args.command == "status":
        pipeline.print_status()
        return 0

    if args.command == "clean":
        pipeline.clean(intermediates=args.intermediates)
        return 0

    return pipeline.build(
        datasets=parse_dataset_selection(args.only),
        keep_raw=args.keep_raw,
        force=args.force,
        from_stage=args.from_stage,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
