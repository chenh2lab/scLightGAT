"""Download the official scGPT whole-human checkpoint from Google Drive."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path

OFFICIAL_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing"
)
REQUIRED_FILES = ("best_model.pt", "vocab.json", "args.json")
INSTALL_ORDER = ("args.json", "vocab.json", "best_model.pt")


def _checkpoint_issue(output_dir, *, deep=False):
    """Return why a checkpoint is unusable, or ``None`` when it is valid."""
    output_dir = Path(output_dir)
    missing = [
        filename
        for filename in REQUIRED_FILES
        if not (output_dir / filename).is_file()
    ]
    if missing:
        return "missing: " + ", ".join(missing)

    for filename in ("vocab.json", "args.json"):
        try:
            with (output_dir / filename).open(encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return f"invalid {filename}: {exc}"
        if not isinstance(value, Mapping) or not value:
            return f"invalid {filename}: expected a non-empty JSON object"

    model_file = output_dir / "best_model.pt"
    if not zipfile.is_zipfile(model_file):
        return "invalid best_model.pt: truncated or not a PyTorch ZIP archive"

    if deep:
        try:
            import torch

            try:
                state = torch.load(
                    model_file, map_location="cpu", weights_only=False
                )
            except TypeError:
                # Compatibility with PyTorch releases predating weights_only.
                state = torch.load(model_file, map_location="cpu")
        except Exception as exc:
            return f"invalid best_model.pt: torch.load failed: {exc}"
        if not isinstance(state, Mapping) or not state:
            return "invalid best_model.pt: expected a non-empty state dict"

    return None


def checkpoint_complete(output_dir):
    return _checkpoint_issue(output_dir) is None


def _download_folder(destination):
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required. Activate the scGPT environment and run: "
            "pip install gdown"
        ) from exc

    try:
        gdown.download_folder(
            OFFICIAL_FOLDER_URL,
            output=str(destination),
            quiet=False,
            use_cookies=False,
            remaining_ok=True,
        )
    except TypeError:
        # Compatibility with older gdown releases without remaining_ok.
        gdown.download_folder(
            OFFICIAL_FOLDER_URL,
            output=str(destination),
            quiet=False,
            use_cookies=False,
        )


def _collect_downloaded_files(staging_dir):
    """Normalize gdown versions that recreate the Drive folder hierarchy."""
    for filename in REQUIRED_FILES:
        destination = staging_dir / filename
        if destination.is_file():
            continue
        matches = [
            path
            for path in staging_dir.rglob(filename)
            if path.is_file() and path != destination
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Checkpoint download did not produce one {filename}; "
                f"found {len(matches)}"
            )
        os.replace(matches[0], destination)


def download_checkpoint(output_dir):
    """Ensure that a complete official whole-human checkpoint is available."""
    output_dir = Path(output_dir)
    issue = _checkpoint_issue(output_dir)
    if issue is None:
        print(f"scGPT checkpoint already exists and is valid: {output_dir}")
        return output_dir

    print(f"scGPT checkpoint requires download ({issue}).")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.download-",
            dir=output_dir.parent,
        )
    )
    try:
        print("Downloading the official scGPT whole-human checkpoint...")
        _download_folder(staging_dir)
        _collect_downloaded_files(staging_dir)

        issue = _checkpoint_issue(staging_dir, deep=True)
        if issue is not None:
            raise RuntimeError("Downloaded checkpoint is incomplete: " + issue)

        output_dir.mkdir(parents=True, exist_ok=True)
        # Publish the model last. If interrupted, the next run still sees an
        # absent or invalid model and retries the complete staged download.
        for filename in INSTALL_ORDER:
            os.replace(staging_dir / filename, output_dir / filename)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"scGPT checkpoint ready: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).parent / "models" / "scGPT_human",
    )
    arguments = parser.parse_args()
    download_checkpoint(arguments.output_dir)
