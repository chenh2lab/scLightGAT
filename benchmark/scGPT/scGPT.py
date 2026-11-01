"""scGPT fine-tuning cell type annotation benchmark."""

import gc
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from scipy.sparse import csr_matrix, save_npz, load_npz

import model_impl
from utils.dataset import is_count_matrix
from utils.pipeline import (
    SCORE_CLASSES_KEY,
    SCORE_KEY,
    create_argument_parser,
    run_annotation_pipeline,
)

PREDICTION_CHUNK_SIZE = 2_048

# Fixed to the official annotation tutorial. These values remain attached to
# the shared argparse namespace so existing run signatures stay unchanged.
SCGPT_SETTINGS = {
    "epochs": model_impl.DEFAULT_EPOCHS,
    "batch_size": model_impl.DEFAULT_BATCH_SIZE,
    "lr": model_impl.DEFAULT_LR,
    "schedule_ratio": model_impl.DEFAULT_SCHEDULE_RATIO,
}


def _preprocess_adata(
    adata, vocab, *, raw_counts=None, genes_already_filtered=False,
):
    """Apply the official normalization/log policy, then sparse-safe binning."""
    adata.var["gene_name"] = adata.var_names.astype(str)
    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    if genes_already_filtered:
        if (adata.var["id_in_vocab"] < 0).any():
            raise ValueError("A prefiltered scGPT query contains unknown genes")
    else:
        adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()
    if adata.n_vars == 0:
        raise ValueError("No query genes occur in the scGPT vocabulary")

    if raw_counts is None:
        raw_counts = is_count_matrix(adata.X)
    else:
        raw_counts = bool(raw_counts)
    # Keep the upstream normalization/log1p behavior, then bin with the
    # mathematically equivalent sparse implementation. Upstream scGPT first
    # densifies the complete matrix, which is unnecessarily slow and memory
    # intensive for this benchmark.
    preprocessor = model_impl.Preprocessor(
        normalize_total=1e4,
        result_normed_key=None,
        log1p=raw_counts,
        result_log1p_key=None,
    )
    preprocessor(adata, batch_key=None)
    print(
        f"[scGPT] sparse-safe binning: {adata.n_obs} cells x {adata.n_vars} genes",
        flush=True,
    )
    binned, bin_edges = model_impl.sparse_safe_binning(
        adata.X, model_impl.DEFAULT_N_BINS
    )
    adata.layers["X_binned"] = binned
    adata.obsm["bin_edges"] = bin_edges
    return adata


def _predict_probabilities(model, loader, vocab, device):
    model.eval()
    probabilities = []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for batch_data in loader:
            gene_ids = batch_data["gene_ids"].to(device)
            values = batch_data["values"].to(device)
            padding_mask = gene_ids.eq(vocab[model_impl.PAD_TOKEN])
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(
                    gene_ids,
                    values,
                    src_key_padding_mask=padding_mask,
                    CLS=True,
                )
            probabilities.append(
                torch.softmax(output["cls_output"], dim=1).cpu().numpy()
            )
    return np.concatenate(probabilities, axis=0)


def train_scgpt(adata, seed, run_out_path, args):
    """Fine-tune the pretrained whole-human scGPT classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_impl._preprocess_adata = _preprocess_adata
    print(
        f"[scGPT] full training reference: {adata.n_obs} cells",
        flush=True,
    )
    model_dir = run_out_path / "_model"
    return model_impl._finetune(
        adata,
        args.train_celltype,
        args.scgpt_model_dir,
        model_impl.ensure_dir(model_dir),
        seed,
        args,
        device,
    )


def load_scgpt_model(shared_run_path, seed, args):
    """Rebuild a completed fine-tuned handle for automatic prediction resume."""
    if args.scgpt_model_dir is None:
        raise ValueError("scGPT pretrained model directory is not configured")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model_impl.load_finetuned_handle(
        Path(shared_run_path) / "_model",
        args.scgpt_model_dir,
        device,
        args=args,
        expected_seed=seed,
    )


def _aligned_prediction_chunk_size(batch_size):
    """Choose a memory chunk boundary that never splits an inference batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batches_per_chunk = max(1, PREDICTION_CHUNK_SIZE // batch_size)
    return batches_per_chunk * batch_size


def _stage_query_binning(adata, vocab, temporary_dir, chunk_size):
    """Bin a complete query in row order and stage sparse CSR chunks.

    Binning must finish for every cell before tokenization begins because both
    operations consume NumPy random numbers in the official scGPT pipeline.
    The returned padding width is therefore computed once across the entire
    query, while the matrices themselves remain disk-backed and temporary.
    """
    gene_names = np.asarray(adata.var_names.astype(str))
    gene_mask = np.fromiter(
        (gene in vocab for gene in gene_names),
        dtype=bool,
        count=len(gene_names),
    )
    if not gene_mask.any():
        raise ValueError("No query genes occur in the scGPT vocabulary")

    filtered_query = adata[:, gene_mask]
    filtered_genes = gene_names[gene_mask].tolist()
    gene_ids = np.asarray(vocab(filtered_genes), dtype=int)
    # Match one unchunked preprocessing call: this decision is made once after
    # vocabulary filtering, never independently from a potentially misleading
    # subset of cells.
    raw_counts = is_count_matrix(filtered_query.X)

    staged_chunks = []
    maximum_row_length = 1  # every sequence contains the leading CLS token
    temporary_dir = Path(temporary_dir)
    for chunk_index, start in enumerate(range(0, adata.n_obs, chunk_size)):
        stop = min(start + chunk_size, adata.n_obs)
        processed = _preprocess_adata(
            filtered_query[start:stop].copy(),
            vocab,
            raw_counts=raw_counts,
            genes_already_filtered=True,
        )
        if processed.var["gene_name"].tolist() != filtered_genes:
            raise RuntimeError("scGPT query gene order changed between chunks")

        binned = csr_matrix(processed.layers["X_binned"], copy=True)
        binned.sum_duplicates()
        binned.eliminate_zeros()
        binned.sort_indices()
        row_nonzero = np.diff(binned.indptr)
        if row_nonzero.size:
            maximum_row_length = max(
                maximum_row_length,
                int(row_nonzero.max()) + 1,
            )

        chunk_path = temporary_dir / f"binned_{chunk_index:06d}.npz"
        save_npz(chunk_path, binned, compressed=False)
        staged_chunks.append((chunk_path, stop - start))
        del processed, binned
        gc.collect()

    padded_length = min(maximum_row_length, model_impl.DEFAULT_MAX_SEQ_LEN)
    return staged_chunks, gene_ids, padded_length, raw_counts


def predict_scgpt(adata, model_handle, pred_key, seed, run_out_path, args):
    """Predict query labels and retain softmax probabilities for AUC."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_handle["model"]
    vocab = model_handle["vocab"]
    id_to_type = model_handle["id2type"]
    pad_value = model_handle["pad_value"]

    if adata.n_obs == 0:
        raise ValueError("Query dataset contains no cells")
    chunk_size = _aligned_prediction_chunk_size(args.batch_size)
    if adata.n_obs > chunk_size:
        n_chunks = (adata.n_obs + chunk_size - 1) // chunk_size
        print(
            f"[scGPT] predicting {adata.n_obs} cells in "
            f"{n_chunks} memory chunks"
        )

    model.to(device)
    probability_parts = []
    # TemporaryDirectory removes every staged matrix on normal completion and
    # on Python exceptions. These files are single-call scratch data, never a
    # reusable cache and never accompanied by a fingerprint.
    with tempfile.TemporaryDirectory(prefix="scgpt_query_") as temporary_dir:
        staged, gene_ids, padded_length, raw_counts = _stage_query_binning(
            adata, vocab, temporary_dir, chunk_size,
        )
        print(
            "[scGPT] full-query preprocessing: "
            f"raw_counts={raw_counts}, padded_length={padded_length}",
            flush=True,
        )

        for chunk_index, (chunk_path, n_cells) in enumerate(staged):
            if chunk_index < len(staged) - 1 and n_cells % args.batch_size:
                raise RuntimeError(
                    "scGPT memory chunk split an inference batch"
                )
            binned = load_npz(chunk_path).tocsr()
            tokenized = model_impl._tokenize_matrix(
                binned,
                gene_ids,
                vocab,
                pad_value,
                padded_length=padded_length,
            )
            test_data = {
                "gene_ids": tokenized["genes"],
                "values": tokenized["values"],
                "celltype_labels": torch.zeros(n_cells, dtype=torch.long),
            }
            loader = model_impl._prepare_dataloader(
                test_data,
                args.batch_size,
                shuffle=False,
                num_workers=args.n_jobs,
            )
            probability_parts.append(
                _predict_probabilities(model, loader, vocab, device)
            )
            del binned, tokenized, test_data, loader
            gc.collect()

    probabilities = np.concatenate(probability_parts, axis=0)
    class_names = np.asarray(
        [str(id_to_type[index]) for index in range(len(id_to_type))], dtype=str
    )

    adata.obs[pred_key] = class_names[probabilities.argmax(axis=1)]
    adata.obsm[SCORE_KEY] = probabilities
    adata.uns[SCORE_CLASSES_KEY] = class_names
    adata.uns["annotation_score_type"] = "probabilities"
    return adata


def find_pretrained_model():
    configured = os.environ.get("SCGPT_MODEL_DIR", "").strip()
    candidates = [
        Path(configured) if configured else None,
        Path(__file__).parent / "models" / "scGPT_human",
        Path(__file__).parent / "models" / "whole_human",
    ]
    for candidate in candidates:
        if candidate is not None and all(
            (candidate / filename).exists()
            for filename in ["best_model.pt", "vocab.json", "args.json"]
        ):
            return candidate
    return None


if __name__ == "__main__":
    parser = create_argument_parser("scGPT", include_n_jobs=True)
    parser.add_argument("--scgpt_model_dir", type=Path, default=None)
    parser.set_defaults(**SCGPT_SETTINGS)
    arguments = parser.parse_args()
    if arguments.scgpt_model_dir is None:
        arguments.scgpt_model_dir = find_pretrained_model()
    if arguments.scgpt_model_dir is None:
        raise FileNotFoundError(
            "scGPT checkpoint not found. Run scGPT/download_checkpoint.py first."
        )

    run_annotation_pipeline(
        "scGPT",
        "scgpt_prediction",
        train_scgpt,
        predict_scgpt,
        arguments,
        load_model_fn=load_scgpt_model,
        expression_layers=(),
        share_train_expression=True,
        keep_var_metadata=False,
    )
