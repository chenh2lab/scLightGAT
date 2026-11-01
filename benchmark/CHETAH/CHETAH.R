#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(argparse)
  library(CHETAH)
  library(Matrix)
  library(S4Vectors)
  library(SingleCellExperiment)
})

parser <- ArgumentParser(description = "CHETAH annotation benchmark")
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
args <- parser$parse_args()

MEMORY_LIMIT_BYTES <- 23 * 1024^3
MEMORY_RESERVE_BYTES <- 2 * 1024^3
RSS_ROUNDING_BYTES <- 64 * 1024^2

set.seed(args$seed)
dir.create(args$save_path, recursive = TRUE, showWarnings = FALSE)

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
  matrix <- methods::as(
    Matrix::readMM(file.path(input_dir, "matrix.mtx")),
    "dgCMatrix"
  )
  barcodes <- utils::read.csv(
    file.path(input_dir, "barcodes.csv"),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )$barcode
  features <- utils::read.csv(
    file.path(input_dir, "features.csv"),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )$feature
  metadata <- utils::read.csv(
    file.path(input_dir, "metadata.csv"),
    row.names = 1,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  rownames(matrix) <- features
  colnames(matrix) <- barcodes
  list(matrix = matrix, metadata = metadata, barcodes = barcodes)
}

log_normalize <- function(matrix) {
  if (
    length(matrix@x) > 0L &&
      any(abs(matrix@x - round(matrix@x)) > 1e-6)
  ) {
    return(matrix)
  }
  library_sizes <- Matrix::colSums(matrix)
  library_sizes[library_sizes == 0] <- 1
  normalized <- matrix %*% Matrix::Diagonal(x = 10000 / library_sizes)
  normalized <- methods::as(normalized, "dgCMatrix")
  normalized@x <- log1p(normalized@x)
  dimnames(normalized) <- dimnames(matrix)
  normalized
}

make_query_sce <- function(data) {
  SingleCellExperiment(
    assays = list(logcounts = log_normalize(data$matrix)),
    colData = DataFrame(data$metadata)
  )
}

make_reference_cells <- function(matrix, labels) {
  normalized <- log_normalize(matrix)
  SingleCellExperiment(
    assays = list(logcounts = normalized),
    colData = DataFrame(
      celltypes = as.character(labels),
      row.names = colnames(normalized)
    )
  )
}

chetah_memory_plan <- function(reference, query) {
  genes_used <- min(200, nrow(query))
  label_count <- length(unique(as.character(reference$celltypes)))
  tree_width_bound <- label_count * (label_count + 1) / 2
  largest_reference_type <- max(table(as.character(reference$celltypes)))

  # MeanRef applies a row statistic to each complete reference type, which can
  # materialize the largest sparse type as a dense matrix. The remaining term
  # covers selected-gene correlation buffers and the complete score tree.
  mean_reference_working <- 8 * nrow(reference) * largest_reference_type * 3
  correlation_working <- 8 * genes_used * ncol(reference) * 6
  reference_working <- mean_reference_working + correlation_working
  query_bytes_per_cell <- (
    8 * genes_used * 6 +
      8 * tree_width_bound * 3
  )
  resident <- process_tree_rss_bytes()
  fixed <- resident + MEMORY_RESERVE_BYTES + reference_working
  full_estimate <- fixed + query_bytes_per_cell * ncol(query)

  if (full_estimate <= MEMORY_LIMIT_BYTES) {
    return(list(
      use_chunks = FALSE,
      chunk_size = ncol(query),
      estimate = full_estimate,
      resident = resident
    ))
  }

  if (fixed >= MEMORY_LIMIT_BYTES) {
    # Every chunk rebuilds the same reference tree and mean profiles. If that
    # immutable work already exceeds the cap, chunking cannot help and would
    # only repeat the dominant computation.
    return(list(
      use_chunks = FALSE,
      chunk_size = ncol(query),
      estimate = full_estimate,
      resident = resident
    ))
  }

  available <- max(0, MEMORY_LIMIT_BYTES - fixed)
  chunk_size <- max(1, floor(available / query_bytes_per_cell))
  chunk_size <- as.integer(min(ncol(query), chunk_size))
  list(
    use_chunks = TRUE,
    chunk_size = chunk_size,
    estimate = fixed + query_bytes_per_cell * chunk_size,
    resident = resident
  )
}

if (args$task == "train_reference") {
  data <- load_exchange(args$input_dir)
  if (!(args$label_col %in% colnames(data$metadata))) {
    stop(sprintf(
      "Label column '%s' not found in training metadata",
      args$label_col
    ))
  }
  labels <- as.character(data$metadata[[args$label_col]])
  if (anyNA(labels) || any(!nzchar(labels))) {
    stop("Training labels contain missing or empty values")
  }
  original_cells <- ncol(data$matrix)
  reference <- make_reference_cells(data$matrix, labels)
  data <- NULL
  gc(verbose = FALSE)
  saveRDS(reference, file.path(args$save_path, "reference.rds"))
  cat(sprintf(
    "[CHETAH] full reference: %d cells retained (%d genes)\n",
    original_cells,
    nrow(reference)
  ))
} else {
  if (is.null(args$reference_rds)) {
    stop("--reference_rds is required for prediction")
  }
  reference <- readRDS(args$reference_rds)
  query_data <- load_exchange(args$input_dir)
  query_barcodes <- as.character(query_data$barcodes)
  query <- make_query_sce(query_data)
  query_data <- NULL
  gc(verbose = FALSE)

  common_genes <- intersect(rownames(reference), rownames(query))
  if (length(common_genes) < 2L) {
    stop("Fewer than two genes overlap reference and query")
  }
  reference <- reference[common_genes, , drop = FALSE]
  query <- query[common_genes, , drop = FALSE]
  n_genes <- min(200L, length(common_genes))
  gc(verbose = FALSE)

  plan <- chetah_memory_plan(reference, query)
  cat(sprintf(
    paste0(
      "[CHETAH] prediction mode=%s, query=%d cells, chunk_size=%d, ",
      "estimated_peak=%.2f GiB, process_tree_rss=%.2f GiB\n"
    ),
    if (plan$use_chunks) "memory_fallback" else "full",
    ncol(query),
    plan$chunk_size,
    plan$estimate / 1024^3,
    plan$resident / 1024^3
  ))
  if (plan$estimate > MEMORY_LIMIT_BYTES) {
    warning(
      paste0(
        "The estimated process-tree peak remains above 23 GiB; query ",
        "chunking cannot guarantee the cap"
      ),
      call. = FALSE
    )
  }

  if (!plan$use_chunks) {
    # n_genes remains at the official default of 200 whenever the input has
    # enough genes and is only bounded to prevent CHETAH indexing past a
    # smaller feature set. All other settings retain their installed defaults.
    annotated <- CHETAHclassifier(
      input = query,
      ref_cells = reference,
      n_genes = n_genes
    )
    labels <- as.character(annotated$celltype_CHETAH)
  } else {
    starts <- seq.int(1L, ncol(query), by = plan$chunk_size)
    label_parts <- vector("list", length(starts))
    for (i in seq_along(starts)) {
      first <- starts[[i]]
      last <- if (i < length(starts)) {
        starts[[i + 1L]] - 1L
      } else {
        ncol(query)
      }
      annotated <- CHETAHclassifier(
        input = query[, first:last, drop = FALSE],
        ref_cells = reference,
        n_genes = n_genes
      )
      label_parts[[i]] <- as.character(annotated$celltype_CHETAH)
      rm(annotated)
      gc(verbose = FALSE)
    }
    labels <- unlist(label_parts, use.names = FALSE)
  }
  labels[is.na(labels) | !nzchar(trimws(labels))] <- "Unknown"
  utils::write.csv(
    data.frame(
      barcode = query_barcodes,
      prediction = labels,
      check.names = FALSE
    ),
    file.path(args$save_path, "predictions.csv"),
    row.names = FALSE,
    quote = TRUE
  )
}
