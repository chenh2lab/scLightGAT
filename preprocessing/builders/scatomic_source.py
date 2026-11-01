from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from utils.config import SCATOMIC_ZENODO
from utils.downloader import download_zenodo_matching, extract_archive
from utils.io import export_r_object


def load_preserved_training_metadata(rawdir: Path, auxdir: Path) -> pd.DataFrame:
    """Load preserved scATOMIC reference metadata, never prediction layers."""
    destination = rawdir / "scATOMIC_7419236"
    destination.mkdir(parents=True, exist_ok=True)

    try:
        files = download_zenodo_matching(
            SCATOMIC_ZENODO,
            [
                r"training_metadata",
                r"reference_datasets",
                r"harmonized_results.*metadata",
            ],
            destination,
        )
    except FileNotFoundError:
        files = download_zenodo_matching(
            SCATOMIC_ZENODO,
            [r"metadata|reference"],
            destination,
            allow_all_fallback=True,
        )

    for path in files:
        if path.name.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz")):
            extract_archive(path, destination / "extracted")

    exported_csvs: list[Path] = []
    r_objects = list(destination.rglob("*.RDS")) + list(
        destination.rglob("*.rds")
    )
    for path in r_objects:
        if not re.search(
            r"training_metadata|harmonized_results.*metadata",
            path.name,
            re.IGNORECASE,
        ):
            continue
        outdir = auxdir / "scatomic_export" / path.stem
        # This scATOMIC object is used only for preserved source/reference
        # metadata.  It is valid for it to contain no expression matrix.
        export_r_object(path, outdir, require_matrix=False)
        exported_csvs.extend(outdir.rglob("*.csv"))

    if not exported_csvs:
        exported_csvs = list(destination.rglob("*.csv"))
    if not exported_csvs:
        raise RuntimeError("Unable to export scATOMIC preserved reference metadata")

    frames = []
    for path in exported_csvs:
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        frame["__scatomic_file__"] = str(path)
        frames.append(frame)

    if not frames:
        raise RuntimeError("No readable scATOMIC metadata tables were produced")
    return pd.concat(frames, ignore_index=True, sort=False)


def select_accession_rows(
    metadata: pd.DataFrame,
    accession: str,
    aliases=(),
) -> pd.DataFrame:
    """Restrict the combined scATOMIC metadata to the intended source study.

    Never fall back to the complete pan-cancer metadata table: that behavior
    permits recurrent 10x barcodes to match cells from unrelated studies.
    """
    evidence_columns = [
        column
        for column in metadata.columns
        if re.search(
            r"dataset|accession|study|training",
            str(column),
            re.IGNORECASE,
        )
    ]

    terms = [str(accession), *[str(x) for x in aliases]]
    hit = pd.Series(False, index=metadata.index)
    for column in evidence_columns:
        series = metadata[column].astype(str)
        for term in terms:
            # accession may be embedded; dataset aliases such as Chen/Lee are
            # matched as full values where possible and as bounded text fallback.
            hit |= series.str.fullmatch(re.escape(term), case=False, na=False)
            hit |= series.str.contains(
                rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
                case=False,
                na=False,
                regex=True,
            )

    if not hit.any():
        values = {}
        for column in evidence_columns[:8]:
            values[column] = sorted(
                metadata[column].dropna().astype(str).unique().tolist()
            )[:30]
        raise RuntimeError(
            f"No scATOMIC preserved metadata rows matched {terms}; "
            f"evidence examples={values}"
        )
    return metadata.loc[hit].copy()
