from __future__ import annotations

import re
import shutil
from pathlib import Path

import anndata as ad
import pandas as pd

from builders.common import best_column
from utils.config import (
    SCDIFFUSION_FIGSHARE_ARTICLE,
    SCDIFFUSION_FIGSHARE_VERSION,
)
from utils.downloader import extract_archive, figshare_article_files, resumable_download


# The public Figshare deposit has changed across versions. Query the current
# manifest as well as v2/v1 and merge by filename so an endpoint that exposes
# only files changed in one version cannot silently hide the desired dataset.
def _merged_figshare_manifest() -> list[dict]:
    manifests: list[list[dict]] = []
    errors: list[str] = []

    for version in (None, SCDIFFUSION_FIGSHARE_VERSION, "1"):
        try:
            manifests.append(
                figshare_article_files(SCDIFFUSION_FIGSHARE_ARTICLE, version)
            )
        except Exception as exc:
            label = "current" if version is None else f"v{version}"
            errors.append(f"{label}: {exc}")

    merged: dict[str, dict] = {}
    for manifest in manifests:
        for item in manifest:
            merged.setdefault(item["name"], item)

    if not merged:
        raise RuntimeError(
            "Could not read the scDiffusion Figshare manifest. " + "; ".join(errors)
        )
    return list(merged.values())


def _source_package_names(dataset_name: str) -> tuple[str, ...]:
    if dataset_name == "lung_full":
        return ("human_lung.zip",)
    if dataset_name == "sapiens_full":
        return ("sapiens.zip",)
    raise KeyError(f"Unknown scDiffusion logical dataset: {dataset_name}")


def _direct_h5ad_patterns(dataset_name: str) -> list[str]:
    """Fallback for a future Figshare version that exposes h5ad directly."""
    if dataset_name == "lung_full":
        return [
            r"^Human[_ .-]*PF[_ .-]*Lung.*\.h5ad$",
            r".*PF.*Lung.*\.h5ad$",
            r"^Human.*Lung.*\.h5ad$",
            r"^lung.*\.h5ad$",
        ]
    if dataset_name == "sapiens_full":
        return [
            r"^(?:Homo[_ .-]*)?Sapiens.*\.h5ad$",
            r"^sapiens.*\.h5ad$",
        ]
    raise KeyError(f"Unknown scDiffusion logical dataset: {dataset_name}")


def _extract_package_once(package: Path, extraction_root: Path) -> list[Path]:
    """Extract one official source package safely and idempotently.

    A completion marker prevents repeatedly unpacking multi-GB archives. If a
    prior extraction was interrupted, the partial directory is removed and the
    archive is extracted again rather than trusting a possibly truncated h5ad.
    """
    package = Path(package)
    extraction_root = Path(extraction_root)
    marker = extraction_root / ".extract_complete"

    if marker.exists():
        h5ads = sorted(extraction_root.rglob("*.h5ad"))
        if h5ads:
            return h5ads
        marker.unlink()

    if extraction_root.exists():
        shutil.rmtree(extraction_root)
    extraction_root.mkdir(parents=True, exist_ok=True)

    extract_archive(package, extraction_root)
    h5ads = sorted(extraction_root.rglob("*.h5ad"))
    if not h5ads:
        files = [
            str(path.relative_to(extraction_root))
            for path in sorted(extraction_root.rglob("*"))
            if path.is_file()
        ]
        preview = ", ".join(files[:30])
        raise RuntimeError(
            f"scDiffusion package {package.name} contained no h5ad file. "
            f"Extracted files: {preview or '<none>'}"
        )

    marker.write_text(f"source={package.name}\nsize={package.stat().st_size}\n")
    return h5ads


def download_paper_dataset(rawdir: Path, dataset_name: str) -> list[Path]:
    """Download and unpack the official scDiffusion source package.

    Figshare v2 distributes the human lung and Tabula Sapiens datasets as
    ``human_lung.zip`` and ``sapiens.zip`` rather than as top-level h5ad files.
    Select the named package first, then return every h5ad contained inside it.
    """
    destination = Path(rawdir) / "scDiffusion"
    destination.mkdir(parents=True, exist_ok=True)

    manifest = _merged_figshare_manifest()
    by_casefold = {item["name"].casefold(): item for item in manifest}

    package_item = None
    for name in _source_package_names(dataset_name):
        package_item = by_casefold.get(name.casefold())
        if package_item is not None:
            break

    if package_item is not None:
        package = resumable_download(
            package_item["url"],
            destination / package_item["name"],
            package_item.get("size"),
        )
        extraction_root = destination / "extracted" / package.stem
        h5ads = _extract_package_once(package, extraction_root)
        print(
            f"  {dataset_name}: extracted {package.name}; "
            f"found {len(h5ads)} h5ad file(s)"
        )
        return h5ads

    # Backward/forward-compatible fallback in case a future Figshare version
    # publishes the source h5ad directly instead of wrapping it in a zip.
    selected: dict[str, dict] = {}
    for pattern in _direct_h5ad_patterns(dataset_name):
        regex = re.compile(pattern, re.IGNORECASE)
        for item in manifest:
            if regex.search(item["name"]):
                selected[item["name"]] = item
        if selected:
            break

    if selected:
        return [
            resumable_download(
                item["url"],
                destination / item["name"],
                item.get("size"),
            )
            for item in selected.values()
        ]

    available = ", ".join(sorted(item["name"] for item in manifest))
    expected_packages = ", ".join(_source_package_names(dataset_name))
    raise FileNotFoundError(
        f"{dataset_name}: official scDiffusion source package not found. "
        f"Expected {expected_packages}. Available Figshare files: {available}"
    )


def _candidate_label_columns(obs: pd.DataFrame) -> list[str]:
    preferred = [
        "Celltype_training",
        "celltype",
        "cell_type",
        "cell type",
        "free_annotation",
        "annotation",
    ]
    columns = [column for column in preferred if column in obs.columns]
    fallback = best_column(obs, [r"cell.*type", r"annotation", r"label"])
    if fallback and fallback not in columns:
        columns.append(fallback)
    return columns


def _normalise_label(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


def identify_dataset(
    files: list[Path],
    expected_labels: set[str],
    dataset_name: str,
) -> tuple[Path, str]:
    """Identify the source h5ad by its locked source-label vocabulary.

    Exact cell counts are intentionally not used as assertions. The official
    archive name already establishes provenance, while the label vocabulary
    verifies that the selected h5ad is the intended benchmark object. The
    observed cell count is reported for the reconstruction record only.
    """
    matches: list[tuple[Path, str]] = []
    diagnostics: list[str] = []
    expected_norm = {_normalise_label(label) for label in expected_labels}

    for path in files:
        try:
            data = ad.read_h5ad(path, backed="r")
        except Exception as exc:
            diagnostics.append(f"{path.name}: unreadable ({exc})")
            continue

        try:
            candidate_columns = _candidate_label_columns(data.obs)
            matching_columns: list[str] = []
            for column in candidate_columns:
                observed = {
                    _normalise_label(value)
                    for value in data.obs[column].dropna().unique()
                }
                if observed == expected_norm:
                    matching_columns.append(column)

            if len(matching_columns) == 1:
                matches.append((path, matching_columns[0]))
            elif len(matching_columns) > 1:
                diagnostics.append(
                    f"{path.name}: multiple label columns match exactly: "
                    f"{matching_columns}; n_obs={data.n_obs}, n_vars={data.n_vars}"
                )
                continue

            diagnostics.append(
                f"{path.name}: n_obs={data.n_obs}, n_vars={data.n_vars}, "
                f"candidate_label_columns={candidate_columns}, "
                f"matching_label_columns={matching_columns}"
            )
        finally:
            data.file.close()

    if len(matches) != 1:
        detail = "\n  ".join(diagnostics)
        raise RuntimeError(
            f"{dataset_name}: expected exactly one h5ad inside the official "
            f"scDiffusion source package with the locked source-label vocabulary; "
            f"found {len(matches)}.\n  {detail}"
        )

    path, column = matches[0]
    data = ad.read_h5ad(path, backed="r")
    try:
        print(
            f"  {dataset_name}: selected {path.name}: "
            f"{data.n_obs:,} cells x {data.n_vars:,} genes; "
            f"source label column={column}"
        )
    finally:
        data.file.close()
    return path, column


def load_exact_source(path: Path, label_column: str):
    data = ad.read_h5ad(path)

    # Preserve source IDs before making AnnData observation names unique. Some
    # distributed benchmark h5ads contain duplicated obs_names.
    if not data.obs_names.is_unique:
        data.obs["source_cell_id"] = data.obs_names.astype(str)
        data.obs_names_make_unique()

    # Only an explicitly preserved count layer is acceptable.  The lung h5ad
    # stores library-normalized values in X and is rebuilt from GEO instead.
    expression_source = None
    if "raw_counts" in data.layers:
        data.X = data.layers["raw_counts"].copy()
        expression_source = "layers/raw_counts"
    elif "counts" in data.layers:
        data.X = data.layers["counts"].copy()
        expression_source = "layers/counts"
    else:
        raise RuntimeError(
            f"{path.name}: no raw_counts/counts layer; normalized X cannot be used as counts"
        )

    data.obs["source_celltype"] = data.obs[label_column].astype(str).str.strip().values
    data.uns["reconstruction_source"] = {
        "resource": "scDiffusion official Figshare paper-data deposit",
        "figshare_article": SCDIFFUSION_FIGSHARE_ARTICLE,
        "figshare_version": SCDIFFUSION_FIGSHARE_VERSION,
        "file": path.name,
        "source_label_column": label_column,
        "expression_source": expression_source,
    }
    return data
