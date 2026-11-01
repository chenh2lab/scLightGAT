from __future__ import annotations

import re
import tarfile
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

USER_AGENT = {
    "User-Agent": "scLightGAT-reconstruction/1.0 (+research reproducibility)",
    # Keep byte counts comparable with HTTP Content-Length.
    "Accept-Encoding": "identity",
}
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024


def _remote_size(url: str) -> int | None:
    """Best-effort HTTP object size used to validate cached GEO downloads."""
    try:
        response = requests.head(
            url,
            headers=USER_AGENT,
            timeout=120,
            allow_redirects=True,
        )
        if response.ok:
            value = response.headers.get("Content-Length")
            if value is not None:
                return int(value)
    except requests.RequestException:
        pass
    return None


def _prepare_partial(destination: Path, remote_size: int | None) -> tuple[Path, int]:
    """Recover a previously cached full/partial file without silently trusting it."""
    partial = Path(f"{destination}.part")

    if destination.exists():
        local_size = destination.stat().st_size
        if remote_size is None or local_size == remote_size:
            return partial, -1  # complete cached file

        # A former downloader version could rename an incomplete HTTP response
        # into the final path. Preserve a short file as a resumable .part file.
        if local_size < remote_size:
            if partial.exists():
                if partial.stat().st_size < local_size:
                    partial.unlink()
                    destination.replace(partial)
                else:
                    destination.unlink()
            else:
                destination.replace(partial)
        else:
            destination.unlink()

    offset = partial.stat().st_size if partial.exists() else 0
    if remote_size is not None and offset > remote_size:
        partial.unlink()
        offset = 0
    return partial, offset


def resumable_download(
    url: str,
    destination: Path,
    expected_size: int | None = None,
    chunk_size: int = DOWNLOAD_CHUNK_BYTES,
) -> Path:
    """Download a file with size-validated resume semantics.

    GEO does not provide sizes through our HTML parser, so older code accepted
    any existing final file and did not compare an inferred Content-Length after
    transfer. For multi-GB RDS files that can turn an interrupted download into
    a confusing downstream parser error. This version validates remote byte size
    whenever the server exposes it and resumes short cached files.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    remote_size = expected_size if expected_size is not None else _remote_size(url)
    partial, offset = _prepare_partial(destination, remote_size)
    if offset == -1:
        return destination

    headers = dict(USER_AGENT)
    if offset:
        headers["Range"] = f"bytes={offset}-"

    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=300,
        allow_redirects=True,
    ) as response:
        # Some servers ignore Range. In that case restart instead of appending.
        if offset and response.status_code == 200:
            offset = 0
            mode = "wb"
        else:
            mode = "ab" if offset else "wb"
        response.raise_for_status()

        total_size = remote_size
        if total_size is None:
            content_range = response.headers.get("Content-Range")
            if content_range and "/" in content_range:
                tail = content_range.rsplit("/", 1)[-1]
                if tail.isdigit():
                    total_size = int(tail)
            if total_size is None:
                content_length = response.headers.get("Content-Length")
                if content_length:
                    total_size = offset + int(content_length)

        downloaded = offset
        last_report = 0.0
        with partial.open(mode) as handle:
            for block in response.iter_content(chunk_size):
                if not block:
                    continue
                handle.write(block)
                downloaded += len(block)

                if time.time() - last_report > 2:
                    message = f"  {destination.name}: {downloaded / 1e9:.2f} GB"
                    if total_size:
                        message += (
                            f" / {total_size / 1e9:.2f} GB "
                            f"({100 * downloaded / total_size:.1f}%)"
                        )
                    print(f"\r{message}", end="", flush=True)
                    last_report = time.time()

    actual_size = partial.stat().st_size
    if total_size is not None and actual_size != total_size:
        print()
        raise IOError(
            f"Incomplete download for {destination}: "
            f"{actual_size} bytes received, expected {total_size}. "
            "The .part file was kept for resume."
        )

    # Print an explicit final line rather than leaving the last periodic progress
    # report at e.g. 98.6%, which is misleading even when the transfer completed.
    final = f"  {destination.name}: {actual_size / 1e9:.2f} GB"
    if total_size:
        final += f" / {total_size / 1e9:.2f} GB (100.0%)"
    print(f"\r{final}", flush=True)

    partial.replace(destination)
    return destination


def _geo_bucket(accession: str) -> str:
    return f"{accession[:-3]}nnn"


def _geo_supplementary_url(accession: str) -> str:
    return (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/"
        f"{_geo_bucket(accession)}/{accession}/suppl/"
    )


def list_http_directory(url: str) -> list[tuple[str, str]]:
    response = requests.get(url, headers=USER_AGENT, timeout=120)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    files: list[tuple[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        name = href.split("/")[-1]
        if name and name not in {".", ".."}:
            files.append((name, urljoin(url, href)))
    return files


def download_geo(
    accession: str,
    patterns: list[str],
    destination: Path,
) -> list[Path]:
    """Download GEO supplementary files matching every requested regex."""
    destination.mkdir(parents=True, exist_ok=True)
    listing = list_http_directory(_geo_supplementary_url(accession))

    downloaded: list[Path] = []
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        matches = [item for item in listing if regex.search(item[0])]
        if not matches:
            raise FileNotFoundError(
                f"{accession}: GEO supplementary pattern not found: {pattern}"
            )

        for name, url in matches:
            path = resumable_download(url, destination / name)
            downloaded.append(path)

    return list(dict.fromkeys(downloaded))


def zenodo_files(record_id: str) -> list[dict]:
    response = requests.get(
        f"https://zenodo.org/api/records/{record_id}",
        headers=USER_AGENT,
        timeout=120,
    )
    response.raise_for_status()

    files = []
    for item in response.json().get("files", []):
        links = item.get("links", {})
        files.append(
            {
                "name": item["key"],
                "url": links.get("content") or links.get("self"),
                "size": item.get("size"),
            }
        )
    return files


def download_zenodo_matching(
    record_id: str,
    patterns: list[str],
    destination: Path,
    allow_all_fallback: bool = False,
) -> list[Path]:
    """Download Zenodo files matching one or more regular expressions."""
    destination.mkdir(parents=True, exist_ok=True)
    available = zenodo_files(record_id)

    selected = []
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        selected.extend(item for item in available if regex.search(item["name"]))

    if not selected and allow_all_fallback:
        selected = available
    if not selected:
        raise FileNotFoundError(
            f"Zenodo {record_id}: no files match {patterns}"
        )

    downloaded: list[Path] = []
    unique = {item["name"]: item for item in selected}
    for item in unique.values():
        path = resumable_download(
            item["url"],
            destination / item["name"],
            item.get("size"),
        )
        downloaded.append(path)
    return downloaded


def figshare_article_files(
    article_id: str,
    version: str | None = None,
) -> list[dict]:
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    if version:
        url += f"/versions/{version}"

    response = requests.get(url, headers=USER_AGENT, timeout=120)
    response.raise_for_status()
    return [
        {
            "name": item["name"],
            "url": item["download_url"],
            "size": item.get("size"),
        }
        for item in response.json().get("files", [])
    ]


def download_figshare_matching(
    article_id: str,
    patterns: list[str],
    destination: Path,
    version: str | None = None,
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    available = figshare_article_files(article_id, version)

    downloaded: list[Path] = []
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        for item in available:
            if regex.search(item["name"]):
                downloaded.append(
                    resumable_download(
                        item["url"],
                        destination / item["name"],
                        item.get("size"),
                    )
                )

    if not downloaded:
        names = [item["name"] for item in available]
        raise FileNotFoundError(
            f"Figshare {article_id}: no match for {patterns}. Available: {names}"
        )
    return list(dict.fromkeys(downloaded))


def _safe_target(root: Path, member_name: str) -> Path:
    root = root.resolve()
    target = (root / member_name).resolve()
    if target != root and root not in target.parents:
        raise RuntimeError(f"Unsafe archive member path: {member_name}")
    return target


def _extract_one(path: Path, destination: Path) -> bool:
    name = path.name.lower()
    if name.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(path, "r:*") as archive:
            for member in archive.getmembers():
                _safe_target(destination, member.name)
            archive.extractall(destination)
        return True

    if name.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                _safe_target(destination, member)
            archive.extractall(destination)
        return True

    return False


def extract_archive(path: Path, destination: Path) -> list[Path]:
    """Extract an archive and recursively unpack nested tar/zip archives."""
    path = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    if not _extract_one(path, destination):
        return [path]

    while True:
        extracted_any = False
        for nested in list(destination.rglob("*")):
            if not nested.is_file() or nested.name.startswith("._"):
                continue
            if not nested.name.lower().endswith((".tar", ".tar.gz", ".tgz", ".zip")):
                continue

            marker = Path(f"{nested}.extracted")
            if marker.exists():
                continue

            stem = re.sub(
                r"(?i)(\.tar\.gz|\.tgz|\.tar|\.zip)$",
                "",
                nested.name,
            )
            nested_destination = nested.parent / f"{stem}_extracted"
            _extract_one(nested, nested_destination)
            marker.touch()
            extracted_any = True

        if not extracted_any:
            break

    return list(destination.rglob("*"))
