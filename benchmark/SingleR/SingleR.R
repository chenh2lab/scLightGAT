#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(argparse)
  library(Matrix)
  library(S4Vectors)
  library(SingleCellExperiment)
  library(SingleR)
})

options(warn = 1)

parser <- ArgumentParser(description = "SingleR annotation benchmark")
parser$add_argument(
  "--task",
  required = TRUE,
  choices = c("train_reference", "predict")
)
parser$add_argument("--input_dir", required = TRUE)
parser$add_argument("--reference_rds")
parser$add_argument("--save_path", required = TRUE)
parser$add_argument("--label_col", required = TRUE)
parser$add_argument("--seed", type = "integer", default = 42L)
parser$add_argument("--n_jobs", type = "integer", default = 12L)
args <- parser$parse_args()

if (is.na(args$n_jobs) || args$n_jobs < 1L) {
  stop("--n_jobs must be at least 1", call. = FALSE)
}
BPPARAM <- BiocParallel::MulticoreParam(
  workers = args$n_jobs,
  progressbar = FALSE,
  stop.on.error = TRUE
)

MEMORY_LIMIT_BYTES <- 23 * 1024^3
MEMORY_RESERVE_BYTES <- 2 * 1024^3
RSS_ROUNDING_BYTES <- 64 * 1024^2
UNAVOIDABLE_BASELINE_CHUNK_SIZE <- 10000L

log_line <- function(fmt, ...) {
  cat(sprintf(fmt, ...), "\n", sep = "", file = stdout())
  flush(stdout())
}

read_status_number <- function(pid, field) {
  path <- sprintf("/proc/%d/status", as.integer(pid))
  lines <- tryCatch(
    readLines(path, warn = FALSE),
    error = function(error) character()
  )
  match <- grep(paste0("^", field, ":"), lines, value = TRUE)
  if (length(match) != 1L) return(0)
  value <- suppressWarnings(
    as.numeric(strsplit(trimws(match), "[[:space:]]+")[[1]][2])
  )
  if (is.na(value)) 0 else value
}

process_tree_rss_bytes <- function() {
  current_pid <- Sys.getpid()
  parent_pid <- read_status_number(current_pid, "PPid")
  rss_kib <- read_status_number(current_pid, "VmRSS")
  if (parent_pid > 0) {
    rss_kib <- rss_kib + read_status_number(parent_pid, "VmRSS")
  }
  bytes <- rss_kib * 1024
  ceiling(bytes / RSS_ROUNDING_BYTES) * RSS_ROUNDING_BYTES
}

load_exchange <- function(input_dir) {
  paths <- c(
    matrix = file.path(input_dir, "matrix.mtx"),
    barcodes = file.path(input_dir, "barcodes.csv"),
    features = file.path(input_dir, "features.csv"),
    metadata = file.path(input_dir, "metadata.csv")
  )

  missing_paths <- paths[!file.exists(paths)]
  if (length(missing_paths) > 0L) {
    stop(
      "Missing input file(s): ",
      paste(missing_paths, collapse = ", "),
      call. = FALSE
    )
  }

  expression <- methods::as(
    Matrix::readMM(paths[["matrix"]]),
    "dgCMatrix"
  )
  barcode_table <- utils::read.csv(
    paths[["barcodes"]],
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  feature_table <- utils::read.csv(
    paths[["features"]],
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  metadata <- utils::read.csv(
    paths[["metadata"]],
    row.names = 1,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  if (!("barcode" %in% colnames(barcode_table))) {
    stop("barcodes.csv must contain a 'barcode' column", call. = FALSE)
  }
  if (!("feature" %in% colnames(feature_table))) {
    stop("features.csv must contain a 'feature' column", call. = FALSE)
  }

  barcodes <- as.character(barcode_table$barcode)
  features <- as.character(feature_table$feature)
  if (length(barcodes) != ncol(expression)) {
    stop("Barcode count does not match matrix columns", call. = FALSE)
  }
  if (length(features) != nrow(expression)) {
    stop("Feature count does not match matrix rows", call. = FALSE)
  }
  if (nrow(expression) < 1L || ncol(expression) < 1L) {
    stop("Expression matrix has zero rows or columns", call. = FALSE)
  }
  if (anyNA(barcodes) || any(!nzchar(barcodes)) || anyDuplicated(barcodes)) {
    stop("Barcodes must be non-empty and unique", call. = FALSE)
  }
  if (anyNA(features) || any(!nzchar(features)) || anyDuplicated(features)) {
    stop("Feature names must be non-empty and unique", call. = FALSE)
  }
  if (length(expression@x) > 0L && any(!is.finite(expression@x))) {
    stop("Expression matrix contains non-finite values", call. = FALSE)
  }

  metadata_index <- match(barcodes, rownames(metadata))
  if (anyNA(metadata_index)) {
    stop("Some matrix barcodes are missing from metadata.csv", call. = FALSE)
  }
  metadata <- metadata[metadata_index, , drop = FALSE]
  rownames(metadata) <- barcodes
  rownames(expression) <- features
  colnames(expression) <- barcodes

  list(
    matrix = expression,
    metadata = metadata,
    barcodes = barcodes
  )
}

log_normalize <- function(expression) {
  expression <- methods::as(expression, "dgCMatrix")
  looks_like_counts <- length(expression@x) == 0L || (
    !any(expression@x < 0) &&
      !any(abs(expression@x - round(expression@x)) > 1e-6)
  )
  if (!looks_like_counts) return(expression)

  library_sizes <- Matrix::colSums(expression)
  zero_library <- library_sizes <= 0
  if (any(zero_library)) {
    log_line(
      "[SingleR] warning: %d cells have zero library size",
      sum(zero_library)
    )
    library_sizes[zero_library] <- 1
  }
  normalized <- expression %*% Matrix::Diagonal(
    x = 10000 / library_sizes
  )
  normalized <- methods::as(normalized, "dgCMatrix")
  normalized@x <- log1p(normalized@x)
  dimnames(normalized) <- dimnames(expression)
  normalized
}

make_sce <- function(data) {
  cell_metadata <- S4Vectors::DataFrame(data$metadata)
  rownames(cell_metadata) <- data$barcodes
  SingleCellExperiment::SingleCellExperiment(
    assays = list(logcounts = log_normalize(data$matrix)),
    colData = cell_metadata
  )
}

singler_memory_plan <- function(trained, query) {
  label_count <- length(trained$labels$unique)
  marker_count <- length(trained$markers$unique)
  resident <- process_tree_rss_bytes()

  # The full score/result object remains necessary for official global
  # pruning. Chunking only bounds the temporary classifier result.
  merged_result <- 8 * ncol(query) * (label_count + 4)
  thread_scratch <- 8 * max(1, marker_count) * 16
  fixed <- resident + MEMORY_RESERVE_BYTES + merged_result + thread_scratch
  chunk_bytes_per_cell <- 8 * (3 * label_count + 16)
  full_estimate <- fixed + chunk_bytes_per_cell * ncol(query)

  if (full_estimate <= MEMORY_LIMIT_BYTES) {
    return(list(
      use_chunks = FALSE,
      chunk_size = ncol(query),
      estimate = full_estimate,
      resident = resident
    ))
  }

  if (fixed >= MEMORY_LIMIT_BYTES) {
    # Query batching cannot reduce the immutable parent/model baseline.
    # Bound incremental work without falling into impractical one-cell calls.
    chunk_size <- min(ncol(query), UNAVOIDABLE_BASELINE_CHUNK_SIZE)
    return(list(
      use_chunks = chunk_size < ncol(query),
      chunk_size = as.integer(chunk_size),
      estimate = fixed + chunk_bytes_per_cell * chunk_size,
      resident = resident
    ))
  }

  available <- max(0, MEMORY_LIMIT_BYTES - fixed)
  chunk_size <- max(1, floor(available / chunk_bytes_per_cell))
  chunk_size <- as.integer(min(ncol(query), chunk_size))
  list(
    use_chunks = TRUE,
    chunk_size = chunk_size,
    estimate = fixed + chunk_bytes_per_cell * chunk_size,
    resident = resident
  )
}

classify_in_chunks <- function(query, trained, chunk_size) {
  starts <- seq.int(1L, ncol(query), by = chunk_size)
  score_matrix <- NULL
  labels <- character(ncol(query))
  delta_next <- numeric(ncol(query))

  for (i in seq_along(starts)) {
    first <- starts[[i]]
    last <- if (i < length(starts)) {
      starts[[i + 1L]] - 1L
    } else {
      ncol(query)
    }

    # Core classification retains all installed defaults except pruning.
    # Pruning is deferred until all chunks have been merged because its
    # per-label median/MAD thresholds are global query statistics.
    current <- SingleR::classifySingleR(
      test = query[, first:last, drop = FALSE],
      trained = trained,
      prune = FALSE,
      num.threads = args$n_jobs,
      BPPARAM = BPPARAM
    )
    current_scores <- as.matrix(current$scores)
    if (is.null(score_matrix)) {
      score_matrix <- matrix(
        NA_real_,
        nrow = ncol(query),
        ncol = ncol(current_scores),
        dimnames = list(colnames(query), colnames(current_scores))
      )
    } else if (!identical(colnames(score_matrix), colnames(current_scores))) {
      stop("SingleR score columns changed between chunks", call. = FALSE)
    }

    indices <- first:last
    score_matrix[indices, ] <- current_scores
    labels[indices] <- as.character(current$labels)
    delta_next[indices] <- as.numeric(current$delta.next)
    rm(current, current_scores)
    gc(verbose = FALSE)
  }

  output <- S4Vectors::DataFrame(
    scores = I(score_matrix),
    labels = labels,
    delta.next = delta_next,
    check.names = FALSE
  )
  rownames(output) <- colnames(query)
  S4Vectors::metadata(output)$common.genes <- trained$markers$unique
  S4Vectors::metadata(output)$de.genes <- trained$markers$full
  output$pruned.labels <- output$labels
  output$pruned.labels[SingleR::pruneScores(output)] <- NA_character_
  output
}

write_prediction <- function(prediction, barcodes, save_path) {
  # SingleR's primary annotation is `labels`; `pruned.labels` is retained as
  # the official diagnostic output but does not replace the benchmark call.
  labels <- as.character(prediction$labels)
  labels[is.na(labels) | !nzchar(labels)] <- "Unknown"
  utils::write.csv(
    data.frame(
      barcode = barcodes,
      prediction = labels,
      check.names = FALSE,
      stringsAsFactors = FALSE
    ),
    file.path(save_path, "predictions.csv"),
    row.names = FALSE,
    quote = TRUE
  )

  scores <- as.matrix(prediction$scores)
  utils::write.csv(
    data.frame(
      barcode = barcodes,
      scores,
      check.names = FALSE,
      stringsAsFactors = FALSE
    ),
    file.path(save_path, "scores.csv"),
    row.names = FALSE,
    quote = TRUE
  )
}

run_train_reference <- function() {
  data <- load_exchange(args$input_dir)
  if (!(args$label_col %in% colnames(data$metadata))) {
    stop(sprintf(
      "Label column '%s' not found in training metadata",
      args$label_col
    ), call. = FALSE)
  }
  labels <- as.character(data$metadata[[args$label_col]])
  if (anyNA(labels) || any(!nzchar(labels))) {
    stop("Training labels contain missing or empty values", call. = FALSE)
  }
  if (length(unique(labels)) < 2L) {
    stop("SingleR requires at least two reference labels", call. = FALSE)
  }

  original_cells <- ncol(data$matrix)
  normalized <- log_normalize(data$matrix)
  reference_coldata <- S4Vectors::DataFrame(label = labels)
  rownames(reference_coldata) <- colnames(normalized)
  reference <- SingleCellExperiment::SingleCellExperiment(
    assays = list(logcounts = normalized),
    colData = reference_coldata
  )
  data <- NULL
  gc(verbose = FALSE)

  model_path <- file.path(args$save_path, "reference.rds")
  saveRDS(
    list(reference = reference, label_col = "label"),
    model_path
  )
  log_line(
    "[SingleR] full reference saved: %d cells, %d genes, %d labels -> %s",
    original_cells,
    nrow(reference),
    length(unique(labels)),
    model_path
  )
}

run_predict <- function() {
  if (is.null(args$reference_rds) || !nzchar(args$reference_rds)) {
    stop("--reference_rds is required for prediction", call. = FALSE)
  }
  reference_object <- readRDS(args$reference_rds)
  if (
    !is.list(reference_object) ||
      is.null(reference_object$reference)
  ) {
    stop("Invalid SingleR reference RDS", call. = FALSE)
  }
  reference <- reference_object$reference
  label_col <- reference_object$label_col

  query_data <- load_exchange(args$input_dir)
  query_barcodes <- as.character(query_data$barcodes)
  query <- make_sce(query_data)
  query_data <- NULL
  gc(verbose = FALSE)

  common_genes <- intersect(rownames(query), rownames(reference))
  if (length(common_genes) < 2L) {
    stop("Fewer than two genes overlap reference and query", call. = FALSE)
  }
  reference_labels <- as.character(
    SummarizedExperiment::colData(reference)[[label_col]]
  )

  log_line(
    paste0(
      "[SingleR] version=%s | training official single-cell workflow ",
      "with de.method=wilcox for ",
      "this query: reference=%d cells, query=%d cells, genes=%d, labels=%d"
    ),
    as.character(utils::packageVersion("SingleR")),
    ncol(reference),
    ncol(query),
    length(common_genes),
    length(unique(reference_labels))
  )

  train_formals <- names(formals(SingleR::trainSingleR))
  if ("test.genes" %in% train_formals) {
    # SingleR >= 2.8 uses test.genes internally, matching the current
    # SingleR() implementation without copying a row-subsetted reference.
    trained <- SingleR::trainSingleR(
      ref = reference,
      labels = reference_labels,
      test.genes = rownames(query),
      de.method = "wilcox",
      num.threads = args$n_jobs,
      BPPARAM = BPPARAM
    )
  } else {
    # Compatibility with older SingleR releases, whose SingleR() wrapper
    # explicitly intersected and reordered both matrices before training.
    query <- query[common_genes, , drop = FALSE]
    reference <- reference[common_genes, , drop = FALSE]
    trained <- SingleR::trainSingleR(
      ref = reference,
      labels = reference_labels,
      de.method = "wilcox",
      num.threads = args$n_jobs,
      BPPARAM = BPPARAM
    )
  }
  reference <- NULL
  reference_object <- NULL
  gc(verbose = FALSE)

  plan <- singler_memory_plan(trained, query)
  log_line(
    paste0(
      "[SingleR] prediction mode=%s, chunk_size=%d, ",
      "estimated_peak=%.2f GiB, process_tree_rss=%.2f GiB"
    ),
    if (plan$use_chunks) "memory_fallback" else "full",
    plan$chunk_size,
    plan$estimate / 1024^3,
    plan$resident / 1024^3
  )
  if (plan$estimate > MEMORY_LIMIT_BYTES) {
    warning(
      paste0(
        "The estimated process-tree peak remains above 23 GiB; query ",
        "chunking cannot guarantee the cap"
      ),
      call. = FALSE
    )
  }

  if (plan$use_chunks) {
    prediction <- classify_in_chunks(query, trained, plan$chunk_size)
  } else {
    # Fine-tuning, pruning, exact neighbor search and all classifier settings
    # remain at their installed defaults; only the CPU resource allowance is
    # supplied explicitly.
    prediction <- SingleR::classifySingleR(
      test = query,
      trained = trained,
      num.threads = args$n_jobs,
      BPPARAM = BPPARAM
    )
  }

  write_prediction(prediction, query_barcodes, args$save_path)
  log_line(
    "[SingleR] completed: %d predictions -> %s",
    ncol(query),
    args$save_path
  )
}

set.seed(args$seed)
dir.create(args$save_path, recursive = TRUE, showWarnings = FALSE)

if (args$task == "train_reference") {
  run_train_reference()
} else {
  run_predict()
}
