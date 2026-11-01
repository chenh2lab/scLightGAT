from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io as sparse_io
from scipy import sparse


def _strip_serialization_quotes(value) -> str:
    """Remove only surrounding whitespace/quotes added by table writers."""
    text = str(value).strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if "sep" not in kwargs:
        if ".tsv" in path.name or path.name.endswith(".txt.gz") or path.suffix == ".txt":
            kwargs["sep"] = "\t"
        elif ".csv" in path.name:
            kwargs["sep"] = ","
    return pd.read_csv(path, low_memory=False, **kwargs)


def aggregate_duplicate_genes(adata: ad.AnnData, source: Path | str) -> ad.AnnData:
    """Sum count columns that share one model-facing gene name."""
    names = pd.Index([_strip_serialization_quotes(x) for x in adata.var_names])
    adata.var_names = names
    duplicate_count = int(len(names) - names.nunique())
    if not duplicate_count:
        return adata

    codes, unique_names = pd.factorize(names, sort=False)
    membership = sparse.csr_matrix(
        (
            np.ones(len(codes), dtype=np.float32),
            (np.arange(len(codes)), codes),
        ),
        shape=(len(codes), len(unique_names)),
    )

    def aggregate(matrix):
        value = matrix if sparse.issparse(matrix) else sparse.csr_matrix(matrix)
        return (value @ membership).tocsr()

    first = ~names.duplicated(keep="first")
    out = ad.AnnData(
        aggregate(adata.X),
        obs=adata.obs.copy(),
        var=adata.var.iloc[np.flatnonzero(first)].copy(),
    )
    out.var_names = pd.Index(unique_names.astype(str))
    for layer, matrix in adata.layers.items():
        out.layers[layer] = aggregate(matrix)
    out.uns = dict(adata.uns)
    print(
        f"  Aggregated {duplicate_count:,} duplicate gene rows in {Path(source).name}",
        flush=True,
    )
    return out


def _read_wide_tsv_streaming(path: Path, batch_genes: int = 64) -> ad.AnnData:
    """Stream a very wide genes x cells TSV without constructing huge pandas frames.

    Public GEO wide matrices occur in two closely related schemas:

    1. ``gene_column_header<TAB>cell1<TAB>cell2...``
    2. ``cell1<TAB>cell2...`` with the first (gene-name) header omitted.

    GSE141445 uses the second form.  Each data row still contains
    ``gene<TAB>count1<TAB>count2...``.  Detect the schema from the first data
    row instead of blindly discarding the first header field.
    """
    import gzip
    import time

    opener = gzip.open if path.name.endswith(".gz") else open
    matrices = []
    genes: list[str] = []
    cells: list[str] | None = None
    batch_data = []
    batch_rows = []
    batch_cols = []
    batch_count = 0
    total_nnz = 0
    started = time.monotonic()

    def flush_batch() -> None:
        nonlocal batch_data, batch_rows, batch_cols, batch_count
        if batch_count == 0:
            return
        if batch_data:
            data = np.concatenate(batch_data).astype(np.float32, copy=False)
            rows = np.concatenate(batch_rows).astype(np.int32, copy=False)
            cols = np.concatenate(batch_cols).astype(np.int32, copy=False)
            block = sparse.coo_matrix(
                (data, (rows, cols)),
                shape=(batch_count, len(cells)),
                dtype=np.float32,
            ).tocsr()
        else:
            block = sparse.csr_matrix((batch_count, len(cells)), dtype=np.float32)
        matrices.append(block)
        batch_data = []
        batch_rows = []
        batch_cols = []
        batch_count = 0

    def parse_gene_row(line: str, line_number: int) -> tuple[str, np.ndarray]:
        line = line.rstrip("\r\n")
        try:
            gene, payload = line.split("\t", 1)
        except ValueError as exc:
            raise ValueError(
                f"Malformed row {line_number} in {path}: expected gene + values"
            ) from exc
        values = np.fromstring(payload, sep="\t", dtype=np.float32)
        return _strip_serialization_quotes(gene), values

    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        header = handle.readline()
        if not header:
            raise ValueError(f"Empty matrix: {path}")
        header_fields = header.rstrip("\r\n").split("\t")
        if not header_fields or (len(header_fields) == 1 and not header_fields[0]):
            raise ValueError(f"Invalid header in wide TSV matrix: {path}")

        # Read the first actual gene row before deciding whether the header has
        # a dedicated gene-column label.  Some GEO files (notably GSE141445)
        # omit that one header field entirely.
        first_line = None
        first_line_number = None
        for line_number, line in enumerate(handle, start=2):
            if line.rstrip("\r\n"):
                first_line = line
                first_line_number = line_number
                break
        if first_line is None or first_line_number is None:
            raise ValueError(f"No gene rows found in {path}")

        first_gene, first_values = parse_gene_row(first_line, first_line_number)

        if first_values.size == len(header_fields):
            # Header contains cell IDs only; the gene-column header is absent.
            cells = [str(x) for x in header_fields]
            header_mode = "missing gene-column header"
        elif first_values.size == len(header_fields) - 1:
            # Conventional wide table: first header token labels the gene column.
            cells = [str(x) for x in header_fields[1:]]
            header_mode = "explicit gene-column header"
        else:
            raise ValueError(
                f"Cannot reconcile header and first data row in {path}: "
                f"header has {len(header_fields):,} fields but row "
                f"{first_line_number} ({first_gene}) has {first_values.size:,} values. "
                "Expected either the same number (missing gene-column header) "
                "or one fewer value (explicit gene-column header)."
            )

        if not cells or any(x == "" for x in cells):
            raise ValueError(f"Empty cell identifier found in header of {path}")
        if len(set(cells)) != len(cells):
            duplicated = pd.Index(cells)[pd.Index(cells).duplicated()].unique().tolist()
            raise ValueError(
                f"Duplicate cell identifiers in header of {path}; examples={duplicated[:5]}"
            )

        print(
            f"  Parsing {path.name}: {len(cells):,} cells; streaming gene rows... "
            f"[{header_mode}]",
            flush=True,
        )

        def consume(gene: str, values: np.ndarray, line_number: int) -> None:
            nonlocal batch_count, total_nnz, batch_data, batch_rows, batch_cols
            if values.size != len(cells):
                raise ValueError(
                    f"Row {line_number} ({gene}) in {path} has {values.size:,} values; "
                    f"expected {len(cells):,}"
                )

            nz = np.flatnonzero(values)
            if nz.size:
                batch_data.append(values[nz])
                batch_rows.append(np.full(nz.size, batch_count, dtype=np.int32))
                batch_cols.append(nz.astype(np.int32, copy=False))
                total_nnz += int(nz.size)

            genes.append(str(gene))
            batch_count += 1
            if batch_count >= batch_genes:
                flush_batch()

            if len(genes) % 250 == 0:
                elapsed = time.monotonic() - started
                print(
                    f"    parsed {len(genes):,} genes | nnz={total_nnz:,} | elapsed={elapsed/60:.1f} min",
                    flush=True,
                )

        # Consume the buffered first row, then the remainder of the stream.
        consume(first_gene, first_values, first_line_number)
        for line_number, line in enumerate(handle, start=first_line_number + 1):
            if not line.rstrip("\r\n"):
                continue
            gene, values = parse_gene_row(line, line_number)
            consume(gene, values, line_number)

    flush_batch()
    if not genes:
        raise ValueError(f"No gene rows found in {path}")

    print(
        f"  Building sparse matrix from {len(genes):,} genes and {total_nnz:,} non-zero values...",
        flush=True,
    )
    genes_by_cells = sparse.vstack(matrices, format="csr")
    cells_by_genes = genes_by_cells.transpose().tocsr()
    print(
        f"  Parsed {path.name}: X={cells_by_genes.shape[0]:,} cells x {cells_by_genes.shape[1]:,} genes",
        flush=True,
    )
    adata = ad.AnnData(
        cells_by_genes,
        obs=pd.DataFrame(index=pd.Index(cells, dtype=str)),
        var=pd.DataFrame(index=pd.Index(genes, dtype=str)),
    )
    return aggregate_duplicate_genes(adata, path)


def read_wide_counts(
    path: Path,
    sep: str | None = None,
    transpose: bool = True,
    chunksize: int = 128,
) -> ad.AnnData:
    """Read a wide count table with bounded dense-memory use.

    Very wide tab-separated genes x cells matrices are parsed row-by-row with
    NumPy so that hundreds of thousands of cell columns never become a pandas
    DataFrame. CSV and non-transposed inputs retain the pandas fallback.
    """
    path = Path(path)
    if sep is None:
        sep = "," if ".csv" in path.name else "\t"

    if transpose and sep == "\t":
        return _read_wide_tsv_streaming(path)

    if not transpose:
        frame = pd.read_csv(path, sep=sep, index_col=0, low_memory=False)
        adata = ad.AnnData(
            sparse.csr_matrix(frame.to_numpy(dtype=np.float32)),
            obs=pd.DataFrame(index=frame.index.astype(str)),
            var=pd.DataFrame(index=frame.columns.astype(str)),
        )
        return aggregate_duplicate_genes(adata, path)

    matrices = []
    genes: list[str] = []
    cells: list[str] | None = None

    for frame in pd.read_csv(
        path,
        sep=sep,
        index_col=0,
        chunksize=chunksize,
        low_memory=False,
    ):
        current_cells = frame.columns.astype(str).tolist()
        if cells is None:
            cells = current_cells
        elif current_cells != cells:
            raise ValueError(f"Column/header changed while streaming {path}")

        genes.extend(frame.index.astype(str).tolist())
        matrices.append(sparse.csr_matrix(frame.to_numpy(dtype=np.float32)))

    if not matrices or cells is None:
        raise ValueError(f"Empty matrix: {path}")

    genes_by_cells = sparse.vstack(matrices, format="csr")
    cells_by_genes = genes_by_cells.T.tocsr()
    adata = ad.AnnData(
        cells_by_genes,
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=genes),
    )
    return aggregate_duplicate_genes(adata, path)


def read_10x_mtx(
    matrix: Path,
    barcodes: Path,
    genes: Path,
    orientation: str = "genes_x_cells",
    *,
    genes_header: int | None = None,
    gene_name_column: str | int | None = None,
    deduplicate_genes: bool = True,
) -> ad.AnnData:
    """Read a MatrixMarket single-cell matrix with explicit source schema support.

    GEO deposits are often *10x-like* rather than literal Cell Ranger outputs.  In
    particular, some gene annotation tables contain a header row (for example
    GSE140228 uses ``ENSEMBL``/``SYMBOL``), whereas standard 10x
    ``genes.tsv``/``features.tsv`` files do not.  ``genes_header`` and
    ``gene_name_column`` let a dataset builder describe that schema explicitly.

    The returned AnnData is always cells x genes. Duplicate model-facing gene
    names are resolved by summing their count columns.
    """
    values = sparse_io.mmread(matrix).tocsr()

    cell_table = pd.read_csv(barcodes, sep="\t", header=None)
    if cell_table.empty:
        raise ValueError(f"Empty barcode table: {barcodes}")
    cell_names = cell_table.iloc[:, 0].astype(str).str.strip().tolist()

    gene_table = pd.read_csv(genes, sep="\t", header=genes_header)
    if gene_table.empty:
        raise ValueError(f"Empty gene table: {genes}")

    if gene_name_column is None:
        gene_series = (
            gene_table.iloc[:, 1]
            if gene_table.shape[1] > 1
            else gene_table.iloc[:, 0]
        )
    elif isinstance(gene_name_column, int):
        if gene_name_column >= gene_table.shape[1]:
            raise ValueError(
                f"Gene-name column index {gene_name_column} is out of range for "
                f"{genes} with {gene_table.shape[1]} columns"
            )
        gene_series = gene_table.iloc[:, gene_name_column]
    else:
        if gene_name_column not in gene_table.columns:
            raise ValueError(
                f"Gene-name column {gene_name_column!r} not found in {genes}; "
                f"available={list(map(str, gene_table.columns))}"
            )
        gene_series = gene_table[gene_name_column]

    gene_names = gene_series.astype(str).str.strip().tolist()

    if orientation == "genes_x_cells":
        source_gene_count, source_cell_count = values.shape
    elif orientation == "cells_x_genes":
        source_cell_count, source_gene_count = values.shape
    else:
        raise ValueError(
            f"Unsupported 10x orientation {orientation!r}; use genes_x_cells or cells_x_genes"
        )

    # Defensive auto-recovery for a single accidental header row when a builder
    # did not explicitly declare it.  We only do this when the matrix dimension
    # proves there is exactly one extra annotation row and that row looks like a
    # header; arbitrary shape mismatches still fail loudly.
    if len(gene_names) == source_gene_count + 1 and genes_header is None:
        first_row = [str(value).strip().lower() for value in gene_table.iloc[0].tolist()]
        header_tokens = ("gene", "genes", "symbol", "ensembl", "gene_id", "geneid")
        if any(any(token in value for token in header_tokens) for value in first_row):
            gene_table = gene_table.iloc[1:].reset_index(drop=True)
            gene_series = (
                gene_table.iloc[:, 1]
                if gene_name_column is None and gene_table.shape[1] > 1
                else gene_table.iloc[:, 0]
                if gene_name_column is None
                else gene_table.iloc[:, gene_name_column]
                if isinstance(gene_name_column, int)
                else gene_table[gene_name_column]
            )
            gene_names = gene_series.astype(str).str.strip().tolist()
            print(
                f"  Detected and skipped one gene-table header row in {Path(genes).name}",
                flush=True,
            )

    if len(cell_names) != source_cell_count or len(gene_names) != source_gene_count:
        # A few deposits are stored transposed relative to their documentation.
        # Accept that only when the names prove the opposite orientation exactly.
        if len(cell_names) == values.shape[0] and len(gene_names) == values.shape[1]:
            orientation = "cells_x_genes"
            source_cell_count, source_gene_count = values.shape
        elif len(cell_names) == values.shape[1] and len(gene_names) == values.shape[0]:
            orientation = "genes_x_cells"
            source_gene_count, source_cell_count = values.shape
        else:
            raise ValueError(
                f"10x shape mismatch: matrix={values.shape}, "
                f"cells={len(cell_names)}, genes={len(gene_names)}; "
                f"barcodes={Path(barcodes).name}, genes_file={Path(genes).name}"
            )

    if orientation == "genes_x_cells":
        values = values.T.tocsr()
    else:
        values = values.tocsr()

    expected = (len(cell_names), len(gene_names))
    if values.shape != expected:
        raise ValueError(
            f"10x post-parse shape mismatch: X={values.shape}, "
            f"cells={len(cell_names)}, genes={len(gene_names)}"
        )

    out = ad.AnnData(
        values,
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=gene_names),
    )
    return aggregate_duplicate_genes(out, genes) if deduplicate_genes else out


def _sanitize_dataframe_for_h5ad(frame: pd.DataFrame, axis_name: str) -> pd.DataFrame:
    """Return an HDF5-safe copy of AnnData obs/var metadata.

    h5py cannot serialize an object array that mixes Python numbers, strings,
    and pandas missing sentinels.  Public single-cell metadata frequently gains
    exactly that dtype after joins/reindexing.  Preserve native numeric columns
    whenever possible; preserve genuine text columns as text; stringify only
    truly heterogeneous object columns as a last-resort, lossless metadata
    representation.
    """
    out = frame.copy()

    # AnnData expects string-like axis names.  Do not let a mixed integer/string
    # index become another HDF5 object-array failure.
    out.index = pd.Index([str(value) for value in out.index], name=out.index.name)

    numeric_types = (int, float, complex, np.integer, np.floating, np.complexfloating)
    bool_types = (bool, np.bool_)

    def is_missing_scalar(value) -> bool:
        if value is None or value is pd.NA:
            return True
        if isinstance(value, (float, np.floating)):
            return bool(np.isnan(value))
        return False

    def stringify(value) -> str:
        if is_missing_scalar(value):
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    for column in out.columns:
        series = out[column]
        if not pd.api.types.is_object_dtype(series.dtype):
            continue

        nonmissing = [value for value in series.tolist() if not is_missing_scalar(value)]

        if not nonmissing:
            # An all-missing object column carries no string-vs-numeric type
            # information.  Float NaN is compact and universally HDF5-safe.
            out[column] = pd.Series(np.nan, index=out.index, dtype=np.float64)
            continue

        if all(isinstance(value, str) for value in nonmissing):
            # Pure textual metadata: retain the exact strings and represent a
            # missing value as the empty string rather than a non-string NA
            # sentinel inside an object array.
            out[column] = series.map(stringify).astype(object)
            continue

        if all(
            isinstance(value, numeric_types) and not isinstance(value, bool_types)
            for value in nonmissing
        ):
            # Numeric source metadata such as nCount_RNA may have become object
            # only because unmatched rows introduced pd.NA.  Restore a numeric
            # dtype; pandas naturally promotes integer+missing to float.
            out[column] = pd.to_numeric(series, errors="coerce")
            continue

        if all(isinstance(value, bool_types) for value in nonmissing):
            # Nullable boolean support varies across AnnData releases.  Use a
            # stable textual representation if missing values are present.
            if len(nonmissing) == len(series):
                out[column] = series.astype(bool)
            else:
                out[column] = series.map(stringify).astype(object)
            continue

        # Truly heterogeneous metadata is unusual but source-backed values
        # should not be discarded merely because HDF5 cannot infer a common
        # scalar dtype.  Convert each value deterministically to text.
        print(
            f"[WARN] {axis_name}[{column!r}] has mixed object types; "
            "serializing values as strings",
            flush=True,
        )
        out[column] = series.map(stringify).astype(object)

    return out


def atomic_write_h5ad(adata, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.h5ad")
    started = time.monotonic()
    print(
        f"[WRITE] {path} ({adata.n_obs:,} cells x {adata.n_vars:,} genes)",
        flush=True,
    )

    # Do not copy X: these matrices can contain hundreds of millions of nonzero
    # values.  Temporarily swap only the small metadata DataFrames, write, then
    # restore the in-memory objects so downstream processing sees unchanged
    # source metadata.
    original_obs = adata.obs
    original_var = adata.var
    adata.obs = _sanitize_dataframe_for_h5ad(original_obs, "obs")
    adata.var = _sanitize_dataframe_for_h5ad(original_var, "var")
    try:
        adata.write_h5ad(temporary, compression="gzip")
    finally:
        adata.obs = original_obs
        adata.var = original_var

    temporary.replace(path)
    print(
        f"[WRITE] complete in {time.monotonic() - started:.1f}s: {path}",
        flush=True,
    )
    return path


def concat_intersection(adatas):
    """Concatenate cells while retaining only genes shared by all inputs."""
    allow_fractional = any(
        bool(adata.uns.get("allow_fractional_raw_counts", False))
        for adata in adatas
    )
    combined = ad.concat(
        adatas,
        join="inner",
        merge="same",
        label=None,
        index_unique=None,
    )
    if allow_fractional:
        combined.uns["allow_fractional_raw_counts"] = True
    return combined


def rscript_available() -> bool:
    return shutil.which("Rscript") is not None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_r_object(
    r_file: Path,
    output_directory: Path,
    selected_columns: Path | None = None,
    *,
    require_matrix: bool = True,
) -> Path:
    """Export an R object through Rscript.

    By default at least one MatrixMarket artifact is required. Set
    ``require_matrix=False`` for metadata-only sources such as the annotated
    GSE153935 Seurat ``.Rdata``: expression comes from the published count
    matrix, while the R object is used only for source-backed cell metadata.

    ``selected_columns`` is an optional newline-delimited cell-ID file. For the
    855k-cell GSE210347 source this lets R subset the sparse matrix before
    MatrixMarket export, avoiding a very large full-matrix text conversion.
    The R helper also detects nested-gzip RDS files used by GEO.
    """
    if not rscript_available():
        raise RuntimeError(
            "Rscript is required for this source. Create the Conda environment "
            "from requirements.txt before reconstruction."
        )

    r_file = Path(r_file)
    output_directory = Path(output_directory)
    selected_columns = Path(selected_columns) if selected_columns else None

    request_parts = [
        str(r_file.resolve()),
        str(r_file.stat().st_size),
        str(r_file.stat().st_mtime_ns),
        f"require_matrix={int(require_matrix)}",
    ]
    if selected_columns is not None:
        request_parts.extend(
            [str(selected_columns.resolve()), _sha256_file(selected_columns)]
        )
    request_key = hashlib.sha256("\n".join(request_parts).encode()).hexdigest()
    marker = output_directory / ".export_request_key"
    complete = output_directory / ".export_complete"

    existing_mtx = list(output_directory.glob("*.mtx")) if output_directory.exists() else []
    existing_csv = list(output_directory.rglob("*.csv")) if output_directory.exists() else []
    required_artifacts_present = bool(existing_mtx) if require_matrix else bool(existing_csv)
    if (
        marker.exists()
        and complete.exists()
        and required_artifacts_present
        and marker.read_text().strip() == request_key
    ):
        print(f"  Reusing cached R export: {output_directory}", flush=True)
        return output_directory

    # Never mix files from a failed/older export with the next attempt.
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve().parent / "r_export.R"
    command = [
        "Rscript",
        str(script),
        str(r_file),
        str(output_directory),
    ]
    if selected_columns is not None:
        command.append(str(selected_columns))

    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"R export failed for {r_file}\n"
            f"STDOUT:\n{process.stdout[-3000:]}\n"
            f"STDERR:\n{process.stderr[-6000:]}"
        )

    matrix_files = list(output_directory.glob("*.mtx"))
    metadata_files = list(output_directory.rglob("*.csv"))
    if not complete.exists():
        raise RuntimeError(
            f"R export for {r_file} did not complete. "
            f"STDOUT:\n{process.stdout[-3000:]}\n"
            f"STDERR:\n{process.stderr[-6000:]}"
        )

    if require_matrix and not matrix_files:
        raise RuntimeError(
            f"R export for {r_file} finished without a matrix artifact. "
            f"STDOUT:\n{process.stdout[-3000:]}\n"
            f"STDERR:\n{process.stderr[-6000:]}"
        )

    if not require_matrix and not metadata_files:
        raise RuntimeError(
            f"R metadata export for {r_file} finished without any CSV metadata artifact. "
            f"STDOUT:\n{process.stdout[-3000:]}\n"
            f"STDERR:\n{process.stderr[-6000:]}"
        )

    for matrix_file in matrix_files:
        stem = matrix_file.with_suffix("")
        row_file = Path(f"{stem}.rows.txt")
        col_file = Path(f"{stem}.cols.txt")
        if not row_file.exists() or not col_file.exists():
            raise RuntimeError(
                f"Incomplete R export artifacts for {matrix_file.name}: "
                f"rows={row_file.exists()}, cols={col_file.exists()}"
            )

    marker.write_text(request_key + "\n")
    if process.stdout.strip():
        print(process.stdout.strip(), flush=True)
    if process.stderr.strip():
        # R package-version warnings are useful diagnostics but are not failures.
        print(process.stderr.strip(), flush=True)
    return output_directory
