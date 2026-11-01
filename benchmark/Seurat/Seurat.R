#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(argparse)
  library(Matrix)
  library(Seurat)
  library(SeuratObject)
})

parser <- ArgumentParser(description = "Seurat label-transfer benchmark")
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

set.seed(args$seed)
dir.create(args$save_path, recursive = TRUE, showWarnings = FALSE)

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

if (args$task == "train_reference") {
  data <- load_exchange(args$input_dir)
  if (!(args$label_col %in% colnames(data$metadata))) {
    stop(sprintf(
      "Label column '%s' not found in training metadata",
      args$label_col
    ))
  }

  # Standard preprocessing from Seurat's official reference-mapping example.
  reference <- CreateSeuratObject(
    counts = data$matrix,
    meta.data = data$metadata
  )
  reference <- NormalizeData(reference)
  reference <- FindVariableFeatures(reference)
  reference <- ScaleData(reference)
  data <- NULL
  gc(verbose = FALSE)

  saveRDS(reference, file.path(args$save_path, "reference.rds"))
  cat(sprintf(
    "[Seurat] full reference: %d cells x %d genes\n",
    ncol(reference),
    nrow(reference)
  ))
} else {
  if (is.null(args$reference_rds)) {
    stop("--reference_rds is required for prediction")
  }
  reference <- readRDS(args$reference_rds)
  if (!(args$label_col %in% colnames(reference@meta.data))) {
    stop(sprintf(
      "Label column '%s' not found in reference metadata",
      args$label_col
    ))
  }

  query_data <- load_exchange(args$input_dir)
  query_barcodes <- as.character(query_data$barcodes)
  query <- CreateSeuratObject(
    counts = query_data$matrix,
    meta.data = query_data$metadata
  )
  query <- NormalizeData(query)
  query <- FindVariableFeatures(query)
  query <- ScaleData(query)
  query_data <- NULL
  gc(verbose = FALSE)

  reference_labels <- as.character(reference@meta.data[[args$label_col]])
  names(reference_labels) <- rownames(reference@meta.data)

  cat(sprintf(
    "[Seurat] mapping full query: %d cells (no chunk fallback)\n",
    ncol(query)
  ))
  anchors <- FindTransferAnchors(
    reference = reference,
    query = query
  )
  prediction <- TransferData(
    anchorset = anchors,
    refdata = reference_labels
  )

  if (!identical(rownames(prediction), query_barcodes)) {
    order <- match(query_barcodes, rownames(prediction))
    if (anyNA(order)) {
      stop("Seurat prediction cells do not match the query")
    }
    prediction <- prediction[order, , drop = FALSE]
  }

  labels <- as.character(prediction$predicted.id)
  labels[is.na(labels) | labels == ""] <- "Unknown"
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

  reference_levels <- unique(reference_labels)
  predicted_id_column <- match("predicted.id", colnames(prediction))
  if (!is.na(predicted_id_column) && length(reference_levels) > 0L) {
    score_indices <- predicted_id_column + seq_along(reference_levels)
    if (max(score_indices) > ncol(prediction)) {
      stop("Seurat returned fewer score columns than reference labels")
    }
    scores <- as.data.frame(
      prediction[, score_indices, drop = FALSE],
      check.names = FALSE
    )
    colnames(scores) <- reference_levels
    scores <- cbind(barcode = query_barcodes, scores)
    utils::write.csv(
      scores,
      file.path(args$save_path, "scores.csv"),
      row.names = FALSE,
      quote = TRUE
    )
  }
}
