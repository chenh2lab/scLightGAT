args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: r_export.R input outdir [selected_cell_ids]")
}

input_file <- args[1]
output_dir <- args[2]
selected_ids_file <- if (length(args) >= 3 && nzchar(args[3])) args[3] else NA_character_
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(Matrix)
  library(SeuratObject)
  library(Seurat)
})

read_rds_robust <- function(path) {
  # GEO supplementary files named *.Rds.gz can contain an outer transport gzip
  # around an RDS that was already compressed by saveRDS(). readRDS(path) opens
  # one compressed connection; in the double-gzip case it then sees gzip bytes
  # instead of an R serialization header and reports "unknown input format".
  # Detect the bytes *after one decompression* and add a second gzip connection
  # only when necessary.
  probe <- gzfile(path, open = "rb")
  magic <- readBin(probe, what = "raw", n = 2)
  close(probe)

  if (length(magic) == 2 && identical(as.integer(magic), c(31L, 139L))) {
    message("Detected nested gzip RDS; unwrapping two compression layers")
    outer <- gzfile(path, open = "rb")
    inner <- gzcon(outer)
    on.exit(try(close(inner), silent = TRUE), add = TRUE)
    return(readRDS(inner))
  }

  readRDS(path)
}

selected_ids <- NULL
if (!is.na(selected_ids_file)) {
  selected_ids <- unique(readLines(selected_ids_file, warn = FALSE))
  selected_ids <- selected_ids[nzchar(selected_ids)]
  message(sprintf("Requested %s selected expression columns", format(length(selected_ids), big.mark = ",")))
}

subset_matrix_columns <- function(matrix, name) {
  if (is.null(selected_ids)) {
    return(matrix)
  }
  ids <- colnames(matrix)
  if (is.null(ids)) {
    stop(paste("matrix lacks column names required for CAF cell selection:", name))
  }

  positions <- match(selected_ids, ids)
  keep <- !is.na(positions)
  if (!any(keep)) {
    stop(paste("none of the selected CAF cell IDs matched matrix columns:", name))
  }

  matched_ids <- selected_ids[keep]
  positions <- positions[keep]
  message(sprintf(
    "%s: selected %s/%s requested cells from %s source columns",
    name,
    format(length(positions), big.mark = ","),
    format(length(selected_ids), big.mark = ","),
    format(ncol(matrix), big.mark = ",")
  ))
  out <- matrix[, positions, drop = FALSE]
  colnames(out) <- matched_ids
  out
}

objects <- list()
if (grepl("\\.rdata(\\.gz)?$", input_file, ignore.case = TRUE)) {
  # load() also supports compressed file paths through R's connection handling.
  environment <- new.env()
  names_loaded <- load(input_file, envir = environment)
  for (name in names_loaded) {
    objects[[name]] <- environment[[name]]
  }
} else {
  objects[["object"]] <- read_rds_robust(input_file)
}

safe_name <- function(value) {
  gsub("[^A-Za-z0-9_.-]", "_", value)
}

write_data_frame <- function(frame, name) {
  row_ids <- rownames(frame)
  if (is.null(row_ids)) {
    row_ids <- as.character(seq_len(nrow(frame)))
  }

  output <- data.frame(
    `__rowname__` = row_ids,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  for (column in names(frame)) {
    values <- frame[[column]]
    if (is.factor(values)) {
      values <- as.character(values)
    }
    if (is.atomic(values) && length(values) == nrow(frame)) {
      output[[column]] <- values
    }
  }

  write.csv(
    output,
    file.path(output_dir, paste0(safe_name(name), ".csv")),
    row.names = FALSE
  )
}

write_matrix <- function(matrix, name) {
  matrix <- subset_matrix_columns(matrix, name)
  if (!inherits(matrix, "sparseMatrix")) {
    matrix <- Matrix(as.matrix(matrix), sparse = TRUE)
  }

  row_ids <- rownames(matrix)
  col_ids <- colnames(matrix)
  if (is.null(row_ids) || is.null(col_ids)) {
    stop(paste("matrix lacks row/column names:", name))
  }

  output_name <- safe_name(name)
  writeMM(matrix, file.path(output_dir, paste0(output_name, ".mtx")))
  writeLines(row_ids, file.path(output_dir, paste0(output_name, ".rows.txt")))
  writeLines(col_ids, file.path(output_dir, paste0(output_name, ".cols.txt")))
}

walk_object <- function(value, name, depth = 0) {
  if (is.data.frame(value)) {
    write_data_frame(value, name)
    return(invisible(NULL))
  }

  if (inherits(value, "sparseMatrix") || is.matrix(value)) {
    write_matrix(value, name)
    return(invisible(NULL))
  }

  if (inherits(value, "Seurat")) {
    write_data_frame(value[[]], paste0(name, "_metadata"))
    return(invisible(NULL))
  }

  if (isS4(value) && "meta.data" %in% slotNames(value)) {
    write_data_frame(
      as.data.frame(slot(value, "meta.data")),
      paste0(name, "_metadata")
    )
    return(invisible(NULL))
  }

  if (is.list(value) && depth < 2 && length(value) < 500) {
    child_names <- names(value)
    if (is.null(child_names)) {
      child_names <- seq_along(value)
    }
    for (index in seq_along(value)) {
      try(
        walk_object(
          value[[index]],
          paste0(name, "_", child_names[index]),
          depth + 1
        ),
        silent = TRUE
      )
    }
  }

  invisible(NULL)
}

for (name in names(objects)) {
  walk_object(objects[[name]], name)
}

writeLines("ok", file.path(output_dir, ".export_complete"))
