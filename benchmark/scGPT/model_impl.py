"""scGPT cell type annotation benchmark.

Fine-tune the whole-human pretrained scGPT model with a CLS (cell-type
classification) objective, following the official tutorial:
  https://scgpt.readthedocs.io/en/latest/tutorial_annotation.html
"""

import gc
import json
import os
import random
import shutil
import sys
import time
import types
import warnings
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ============================================================
# torchtext compatibility shim
# ============================================================
# torchtext 0.18 (the last release) ships a C extension compiled against
# torch ≤2.3.  When a newer torch is installed the shared library fails to
# load (undefined symbol).  scGPT only uses torchtext for the Vocab class,
# so we provide a lightweight pure-Python replacement when the native
# extension is broken.

def _install_torchtext_shim() -> None:
    """Replace torchtext's C-backed Vocab with a pure-Python version."""
    try:                       # fast path: torchtext loads normally
        import torchtext       # noqa: F401
        return
    except (OSError, ImportError):
        pass
    # Remove partially-loaded torchtext submodules
    for key in [k for k in sys.modules if k == "torchtext" or k.startswith("torchtext.")]:
        del sys.modules[key]

    import torch.nn as _nn

    # --- Pure-Python VocabPybind replacement ---
    class _VocabPybind:
        def __init__(self, tokens, default_index=None):
            self.itos_ = list(tokens) if tokens else []
            self.stoi_ = {t: i for i, t in enumerate(self.itos_)}
            self.default_index_ = default_index

        def __len__(self):
            return len(self.itos_)

        def __contains__(self, token):
            return token in self.stoi_

        def __getitem__(self, token):
            if token in self.stoi_:
                return self.stoi_[token]
            if self.default_index_ is not None:
                return self.default_index_
            raise RuntimeError(f"Token '{token}' not found and default index is not set")

        def set_default_index(self, index):
            self.default_index_ = index

        def get_default_index(self):
            return self.default_index_

        def insert_token(self, token, index):
            if token in self.stoi_:
                raise RuntimeError(f"Token '{token}' already exists")
            if index < 0 or index > len(self.itos_):
                raise RuntimeError(f"Index {index} not in [0, {len(self.itos_)}]")
            self.itos_.insert(index, token)
            self.stoi_ = {t: i for i, t in enumerate(self.itos_)}

        def append_token(self, token):
            if token in self.stoi_:
                raise RuntimeError(f"Token '{token}' already exists")
            self.stoi_[token] = len(self.itos_)
            self.itos_.append(token)

        def lookup_token(self, index):
            return self.itos_[index]

        def lookup_tokens(self, indices):
            return [self.itos_[i] for i in indices]

        def lookup_indices(self, tokens):
            return [self[t] for t in tokens]

        def get_stoi(self):
            return dict(self.stoi_)

        def get_itos(self):
            return list(self.itos_)

    # --- Pure-Python Vocab (mirrors torchtext.vocab.Vocab) ---
    class _Vocab(_nn.Module):
        def __init__(self, vocab):
            super().__init__()
            self.vocab = vocab

        def __len__(self):
            return len(self.vocab)

        def __contains__(self, token):
            return token in self.vocab

        def __getitem__(self, token):
            return self.vocab[token]

        def forward(self, tokens):
            return self.vocab.lookup_indices(tokens)

        def set_default_index(self, index):
            self.vocab.set_default_index(index)

        def get_default_index(self):
            return self.vocab.get_default_index()

        def insert_token(self, token, index):
            self.vocab.insert_token(token, index)

        def append_token(self, token):
            self.vocab.append_token(token)

        def lookup_token(self, index):
            return self.vocab.lookup_token(index)

        def lookup_tokens(self, indices):
            return self.vocab.lookup_tokens(indices)

        def lookup_indices(self, tokens):
            return self.vocab.lookup_indices(tokens)

        def get_stoi(self):
            return self.vocab.get_stoi()

        def get_itos(self):
            return self.vocab.get_itos()

    def _vocab_factory(ordered_dict, min_freq=1, specials=None, special_first=True):
        specials = specials or []
        for tok in specials:
            ordered_dict.pop(tok, None)
        tokens = [tok for tok, freq in ordered_dict.items() if freq >= min_freq]
        if special_first:
            tokens[0:0] = specials
        else:
            tokens.extend(specials)
        return _Vocab(_VocabPybind(tokens, None))

    def _build_vocab_from_iterator(iterator, min_freq=1, specials=None,
                                    special_first=True, max_tokens=None):
        counter = Counter()
        for tokens in iterator:
            counter.update(tokens)
        specials = specials or []
        sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        if max_tokens is not None:
            sorted_items = sorted_items[: max_tokens - len(specials)]
        od = OrderedDict(sorted_items)
        return _vocab_factory(od, min_freq=min_freq, specials=specials,
                              special_first=special_first)

    def _noop_log(*_a, **_kw):
        pass

    # --- Assemble fake torchtext modules ---
    tt = types.ModuleType("torchtext")
    tt._WARN = False
    tt._TORCHTEXT_DEPRECATION_MSG = ""
    tt.disable_torchtext_deprecation_warning = lambda: None
    sys.modules["torchtext"] = tt

    ext = types.ModuleType("torchtext._extension")
    sys.modules["torchtext._extension"] = ext
    tt._extension = ext

    mu = types.ModuleType("torchtext._internal.module_utils")
    mu.is_module_available = lambda *a, **kw: True
    _internal = types.ModuleType("torchtext._internal")
    _internal.module_utils = mu
    sys.modules["torchtext._internal"] = _internal
    sys.modules["torchtext._internal.module_utils"] = mu
    tt._internal = _internal

    tt_tt = types.ModuleType("torchtext._torchtext")
    tt_tt.Vocab = _VocabPybind
    sys.modules["torchtext._torchtext"] = tt_tt
    tt._torchtext = tt_tt

    utils_mod = types.ModuleType("torchtext.utils")
    utils_mod._log_class_usage = _noop_log
    sys.modules["torchtext.utils"] = utils_mod
    tt.utils = utils_mod

    vocab_mod = types.ModuleType("torchtext.vocab")
    vocab_mod.Vocab = _Vocab
    vocab_mod.vocab = _vocab_factory
    vocab_mod.build_vocab_from_iterator = _build_vocab_from_iterator
    sys.modules["torchtext.vocab"] = vocab_mod
    sys.modules["torchtext.vocab.vocab"] = types.ModuleType("torchtext.vocab.vocab")
    sys.modules["torchtext.vocab.vocab"].Vocab = _Vocab
    sys.modules["torchtext.vocab.vocab_factory"] = types.ModuleType("torchtext.vocab.vocab_factory")
    sys.modules["torchtext.vocab.vocab_factory"].vocab = _vocab_factory
    sys.modules["torchtext.vocab.vocab_factory"].build_vocab_from_iterator = _build_vocab_from_iterator
    tt.vocab = vocab_mod

_install_torchtext_shim()

# scGPT 0.2.4 imports IPython only to detect notebooks but does not declare it
# as a package dependency. Keep command-line runs usable in minimal envs.
try:
    import IPython  # noqa: F401
except ImportError:
    ipython_mod = types.ModuleType("IPython")
    ipython_mod.get_ipython = lambda: None
    ipython_mod.version_info = (0, 0)
    sys.modules["IPython"] = ipython_mod

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, issparse
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader

from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab, pad_batch
from scgpt.preprocess import Preprocessor, _digitize
from scgpt.utils import set_seed

from label_utils import get_broad_type
from utils.dataset import is_count_matrix
from scgpt_compat import (
    save_subtype_results, load_adata, ensure_dir,
    resolve_label_col,
    BROAD_TYPES_FOR_SUBTYPE,
)

warnings.filterwarnings("ignore")

# ============================================================
# Benchmark CLI defaults from the official annotation tutorial.
# Resource safeguards do not override these training hyperparameters.
# ============================================================

DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-4
DEFAULT_SCHEDULE_RATIO = 0.9
DEFAULT_MAX_SEQ_LEN = 3001
DEFAULT_N_BINS = 51
DEFAULT_DROPOUT = 0.2

TRAINING_STATE_FILE = "training_state.pt"
TRAINING_STATE_SCHEMA = 4
REQUESTED_USE_FAST_TRANSFORMER = True
REQUESTED_FAST_TRANSFORMER_BACKEND = "flash"
PRETRAINED_LOAD_POLICY = "official-backend-aware-qkv-strict-audit-v2"

PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]


# ============================================================
# Dataset
# ============================================================

class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# ============================================================
# Preprocessing
# ============================================================

def _preprocess_adata(adata, vocab):
    """Apply the official normalization/log policy, then sparse-safe binning."""
    adata.var["gene_name"] = adata.var.index.tolist()
    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()
    if adata.n_vars == 0:
        raise ValueError("No query genes occur in the scGPT vocabulary")

    raw_counts = is_count_matrix(adata.X)
    preprocessor = Preprocessor(
        normalize_total=1e4,
        result_normed_key=None,
        log1p=raw_counts,
        result_log1p_key=None,
    )
    preprocessor(adata, batch_key=None)
    binned, bin_edges = sparse_safe_binning(adata.X, DEFAULT_N_BINS)
    adata.layers["X_binned"] = binned
    adata.obsm["bin_edges"] = bin_edges
    return adata


def sparse_safe_binning(matrix, n_bins, progress_interval=2000):
    """Reproduce scGPT's row-wise binning without densifying sparse input.

    The upstream preprocessor converts the entire expression matrix to a dense
    array before iterating over cells. This implementation uses the same
    ``np.quantile`` and scGPT ``_digitize`` calls, in the same row order, so
    it consumes the same random numbers and produces the same binned values
    while visiting only stored non-zero entries for sparse matrices.
    """
    if not isinstance(n_bins, int):
        raise ValueError(f"Binning arg must be an integer, but got {n_bins}.")

    quantiles = np.linspace(0, 1, n_bins - 1)
    bin_edges = []
    all_zero_rows = 0

    if issparse(matrix):
        source = matrix.tocsr(copy=True)
        source.sum_duplicates()
        source.eliminate_zeros()
        source.sort_indices()
        if source.data.size and source.data.min() < 0:
            raise ValueError(
                f"Assuming non-negative data, but got min value {source.data.min()}."
            )

        binned_data = np.empty(source.data.shape, dtype=np.int64)
        for row_index in range(source.shape[0]):
            first = source.indptr[row_index]
            last = source.indptr[row_index + 1]
            non_zero_row = source.data[first:last]
            if non_zero_row.size == 0:
                all_zero_rows += 1
                bin_edges.append(np.array([0] * n_bins))
            else:
                bins = np.quantile(non_zero_row, quantiles)
                binned_data[first:last] = _digitize(non_zero_row, bins)
                bin_edges.append(np.concatenate([[0], bins]))
            if progress_interval and (row_index + 1) % progress_interval == 0:
                print(
                    f"[scGPT] binned {row_index + 1}/{source.shape[0]} cells",
                    flush=True,
                )

        binned = csr_matrix(
            (binned_data, source.indices.copy(), source.indptr.copy()),
            shape=source.shape,
        )
    else:
        source = np.asarray(matrix)
        if source.size and source.min() < 0:
            raise ValueError(
                f"Assuming non-negative data, but got min value {source.min()}."
            )
        binned = np.zeros(source.shape, dtype=np.int64)
        for row_index, row in enumerate(source):
            non_zero_ids = row.nonzero()
            non_zero_row = row[non_zero_ids]
            if non_zero_row.size == 0:
                all_zero_rows += 1
                bin_edges.append(np.array([0] * n_bins))
            else:
                bins = np.quantile(non_zero_row, quantiles)
                binned[row_index, non_zero_ids[0]] = _digitize(
                    non_zero_row, bins
                )
                bin_edges.append(np.concatenate([[0], bins]))
            if progress_interval and (row_index + 1) % progress_interval == 0:
                print(
                    f"[scGPT] binned {row_index + 1}/{source.shape[0]} cells",
                    flush=True,
                )

    if all_zero_rows:
        warnings.warn(
            f"The input data contains {all_zero_rows} all-zero rows; "
            "their binned values remain zero."
        )
    edges = np.stack(bin_edges) if bin_edges else np.empty((0, n_bins))
    return binned, edges


def _pad_tokenized_rows(tokenized_rows, padded_length, vocab, pad_value):
    """Pad rows to a full-query length while matching scGPT RNG calls."""
    if padded_length < 1 or padded_length > DEFAULT_MAX_SEQ_LEN:
        raise ValueError(
            f"padded_length must be in [1, {DEFAULT_MAX_SEQ_LEN}]"
        )

    pad_id = vocab[PAD_TOKEN]
    gene_rows = []
    value_rows = []
    for genes, values, _ in tokenized_rows:
        if len(genes) > padded_length:
            indices = np.random.choice(
                len(genes) - 1, padded_length - 1, replace=False,
            )
            indices = np.insert(indices + 1, 0, 0)
            genes = genes[indices]
            values = values[indices]
        if len(genes) < padded_length:
            genes = torch.cat([
                genes,
                torch.full(
                    (padded_length - len(genes),),
                    pad_id,
                    dtype=genes.dtype,
                ),
            ])
            values = torch.cat([
                values,
                torch.full(
                    (padded_length - len(values),),
                    pad_value,
                    dtype=values.dtype,
                ),
            ])
        gene_rows.append(genes)
        value_rows.append(values)

    return {
        "genes": torch.stack(gene_rows, dim=0),
        "values": torch.stack(value_rows, dim=0),
    }


def _tokenize_matrix(
    matrix, gene_ids, vocab, pad_value, *, padded_length=None,
):
    """Tokenize binned data without a full dense intermediate.

    ``padded_length`` is used by query streaming after the maximum sequence
    length has been computed across the complete query. This prevents each
    disk-backed chunk from choosing a different local padding width while
    retaining scGPT's row-order random gene subsampling.
    """
    if not issparse(matrix) and padded_length is None:
        return tokenize_and_pad_batch(
            np.asarray(matrix),
            gene_ids,
            max_len=DEFAULT_MAX_SEQ_LEN,
            vocab=vocab,
            pad_token=PAD_TOKEN,
            pad_value=pad_value,
        )

    source = csr_matrix(matrix, copy=True)
    source.sum_duplicates()
    source.eliminate_zeros()
    source.sort_indices()
    cls_id = vocab["<cls>"]
    tokenized_rows = []
    for row_index in range(source.shape[0]):
        first = source.indptr[row_index]
        last = source.indptr[row_index + 1]
        columns = source.indices[first:last]
        values = source.data[first:last]
        genes = gene_ids[columns]

        # Match scGPT.tokenize_batch exactly: prepend CLS, then convert genes
        # to int64 and expression values to float32 torch tensors.
        genes = np.insert(genes, 0, cls_id)
        values = np.insert(values, 0, 0)
        tokenized_rows.append(
            (torch.from_numpy(genes).long(), torch.from_numpy(values).float(), None)
        )

    if padded_length is None:
        return pad_batch(
            tokenized_rows,
            DEFAULT_MAX_SEQ_LEN,
            vocab,
            PAD_TOKEN,
            pad_value,
            cls_appended=True,
        )
    return _pad_tokenized_rows(
        tokenized_rows, padded_length, vocab, pad_value,
    )


def _tokenize(adata, gene_ids, vocab, pad_value):
    """Extract binned counts and tokenize."""
    return _tokenize_matrix(
        adata.layers["X_binned"], gene_ids, vocab, pad_value
    )


def _prepare_dataloader(
    data_pt, batch_size, shuffle=False, num_workers=0,
):
    """Build a loader with an explicit benchmark-wide worker allowance."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    return DataLoader(
        dataset=SeqDataset(data_pt),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# Model Construction & Loading
# ============================================================

def _torch_load(path, map_location):
    """Load trusted local checkpoints across old and new Torch releases."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _atomic_torch_save(payload, path):
    """Write a Torch checkpoint atomically in the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_dump(payload, path):
    """Write JSON atomically so a killed process cannot expose a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with open(temporary, "w") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _attention_backend(model):
    """Return both the requested and resolved attention implementations."""
    encoder = model.transformer_encoder
    layers = getattr(encoder, "layers", None)
    layer = layers[0] if layers is not None and len(layers) else None

    def _class_name(value):
        if value is None:
            return None
        cls = type(value)
        return f"{cls.__module__}.{cls.__qualname__}"

    cuda_backend = torch.backends.cuda

    def _backend_flag(name):
        getter = getattr(cuda_backend, name, None)
        if not callable(getter):
            return None
        try:
            return bool(getter())
        except (RuntimeError, TypeError):
            return None

    effective_fast = bool(getattr(model, "use_fast_transformer", False))
    requested_fast = bool(getattr(
        model,
        "_requested_use_fast_transformer",
        REQUESTED_USE_FAST_TRANSFORMER,
    ))
    requested_backend = str(getattr(
        model,
        "_requested_fast_transformer_backend",
        REQUESTED_FAST_TRANSFORMER_BACKEND,
    ))
    effective_backend = requested_backend if effective_fast else "pytorch"
    return {
        "pretrained_load_policy": PRETRAINED_LOAD_POLICY,
        "requested": {
            "use_fast_transformer": requested_fast,
            "fast_transformer_backend": requested_backend,
        },
        "effective": {
            "use_fast_transformer": effective_fast,
            "backend": effective_backend,
        },
        "encoder_class": _class_name(encoder),
        "encoder_layer_class": _class_name(layer),
        "pytorch_native_sdpa": {
            "flash_enabled": _backend_flag("flash_sdp_enabled"),
            "memory_efficient_enabled": _backend_flag(
                "mem_efficient_sdp_enabled"
            ),
            "math_enabled": _backend_flag("math_sdp_enabled"),
            "cudnn_enabled": _backend_flag("cudnn_sdp_enabled"),
            "flash_available": _backend_flag(
                "is_flash_attention_available"
            ),
        },
    }


def _pretrained_identity(pretrained_dir):
    """Identify the pretrained weights/config used to construct the model."""
    pretrained_dir = Path(pretrained_dir).resolve()
    checkpoint = pretrained_dir / "best_model.pt"
    config = pretrained_dir / "args.json"
    vocab = pretrained_dir / "vocab.json"
    files = {}
    for path in [checkpoint, config, vocab]:
        stat = path.stat()
        files[path.name] = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    return {"directory": str(pretrained_dir), "files": files}


def _training_hyperparameters(args):
    """Return the exact scGPT settings that define a reusable model."""
    return {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "schedule_ratio": float(args.schedule_ratio),
        "max_seq_len": DEFAULT_MAX_SEQ_LEN,
        "n_bins": DEFAULT_N_BINS,
        "dropout": DEFAULT_DROPOUT,
        "preprocessing": (
            "normalize-total-1e4-conditional-log1p-sparse-binning-v1"
        ),
    }


def _load_pretrained_state_strict(model, pretrained_state):
    """Apply scGPT's backend-aware QKV policy and audit retained weights.

    The whole-human checkpoint contains optional pretraining-only modules and
    no task-specific classifier. Those known source-only modules and the new
    classifier are the only missing keys accepted here. Shape mismatches in
    any non-classification parameter are fatal.
    """
    if not isinstance(pretrained_state, dict):
        raise TypeError("scGPT pretrained checkpoint must be a state dictionary")

    use_fast_transformer = bool(
        getattr(model, "use_fast_transformer", False)
    )
    remapped_state = OrderedDict()
    source_for_target = {}
    for source_key, value in pretrained_state.items():
        # This is the exact backend condition used by
        # scgpt.utils.load_pretrained: Flash retains Wqkv parameters, while
        # the official PyTorch fallback maps them to MultiheadAttention's
        # in_proj_* names.
        target_key = source_key
        if not use_fast_transformer:
            target_key = source_key.replace("Wqkv.", "in_proj_")
        if target_key in remapped_state:
            raise RuntimeError(
                "scGPT checkpoint key collision after Wqkv remapping: "
                f"{source_key} -> {target_key}"
            )
        remapped_state[target_key] = value
        source_for_target[target_key] = source_key

    model_state = model.state_dict()
    qkv_marker = (
        ".self_attn.Wqkv."
        if use_fast_transformer
        else ".self_attn.in_proj_"
    )
    expected_qkv = {key for key in model_state if qkv_marker in key}
    mapped_qkv = {
        key for key in remapped_state
        if qkv_marker in key
    }
    if not expected_qkv or mapped_qkv != expected_qkv:
        missing = sorted(expected_qkv - mapped_qkv)
        unexpected = sorted(mapped_qkv - expected_qkv)
        raise RuntimeError(
            "scGPT encoder QKV loading is incomplete for the resolved "
            f"{'Flash' if use_fast_transformer else 'PyTorch'} backend: "
            f"missing={missing}, unexpected={unexpected}"
        )

    known_source_only_prefixes = (
        "flag_encoder.",
        "mvc_decoder.",
    )
    source_only = sorted(
        key for key in remapped_state
        if key not in model_state
    )
    unknown_source_only = [
        key for key in source_only
        if not key.startswith(known_source_only_prefixes)
    ]
    if unknown_source_only:
        raise RuntimeError(
            "Unexpected scGPT checkpoint parameters absent from the model: "
            + ", ".join(unknown_source_only)
        )

    shape_mismatches = {
        key: (tuple(value.shape), tuple(model_state[key].shape))
        for key, value in remapped_state.items()
        if key in model_state and value.shape != model_state[key].shape
    }
    invalid_shape_mismatches = {
        key: shapes for key, shapes in shape_mismatches.items()
        if not key.startswith("cls_decoder.")
    }
    if invalid_shape_mismatches:
        raise RuntimeError(
            "Non-classification scGPT checkpoint shape mismatches: "
            + repr(invalid_shape_mismatches)
        )

    loadable = {
        key: value for key, value in remapped_state.items()
        if key in model_state and value.shape == model_state[key].shape
    }
    target_only = sorted(set(model_state) - set(loadable))
    invalid_target_only = [
        key for key in target_only
        if not key.startswith("cls_decoder.")
    ]
    if invalid_target_only:
        raise RuntimeError(
            "scGPT pretrained checkpoint did not initialize model parameters: "
            + ", ".join(invalid_target_only)
        )

    updated_state = OrderedDict(model_state)
    updated_state.update(loadable)
    model.load_state_dict(updated_state, strict=True)

    loaded_state = model.state_dict()
    unequal_qkv = []
    for target_key in sorted(expected_qkv):
        source_key = source_for_target[target_key]
        if not torch.equal(
            loaded_state[target_key].detach().cpu(),
            pretrained_state[source_key].detach().cpu(),
        ):
            unequal_qkv.append(f"{source_key} -> {target_key}")
    if unequal_qkv:
        raise RuntimeError(
            "scGPT QKV tensor equality audit failed: "
            + ", ".join(unequal_qkv)
        )

    audit = {
        "policy": PRETRAINED_LOAD_POLICY,
        "effective_qkv_layout": (
            "Wqkv" if use_fast_transformer else "in_proj"
        ),
        "loaded_tensors": len(loadable),
        "qkv_tensors": len(expected_qkv),
        "new_classifier_tensors": len(target_only),
        "ignored_pretraining_only_tensors": len(source_only),
    }
    model._pretrained_load_audit = audit
    print(
        "[scGPT] pretrained checkpoint audit: "
        + json.dumps(audit, sort_keys=True),
        flush=True,
    )
    return model


def load_finetuned_handle(
    save_dir, pretrained_dir, device, *, args, expected_seed,
):
    """Rebuild an inference handle from a completed fine-tuned checkpoint."""
    save_dir = Path(save_dir)
    model_file = save_dir / "best_model.pt"
    label_file = save_dir / "label_map.json"
    gene_file = save_dir / "gene_list.json"
    identity_file = save_dir / "model_identity.json"
    required = [model_file, label_file, gene_file, identity_file]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        return None

    with open(identity_file, "r") as handle:
        identity = json.load(handle)
    expected_hyperparameters = _training_hyperparameters(args)
    if identity.get("schema") != TRAINING_STATE_SCHEMA:
        raise RuntimeError(
            "scGPT completed model uses an obsolete training schema"
        )
    if identity.get("seed") != int(expected_seed):
        raise RuntimeError("scGPT completed model seed changed")
    if identity.get("hyperparameters") != expected_hyperparameters:
        raise RuntimeError(
            "scGPT completed model hyperparameters/preprocessing changed"
        )
    if identity.get("pretrained") != _pretrained_identity(pretrained_dir):
        raise RuntimeError("scGPT checkpoint pretrained-model identity changed")
    data_identity = identity.get("data", {})
    if data_identity.get("source") != _training_source_identity(args.train_path):
        raise RuntimeError("scGPT completed model training input changed")

    with open(label_file, "r") as handle:
        raw_id2type = json.load(handle)
    if not isinstance(raw_id2type, dict):
        raise ValueError(f"Invalid scGPT label map: {label_file}")
    id2type = {int(key): str(value) for key, value in raw_id2type.items()}
    if sorted(id2type) != list(range(len(id2type))):
        raise ValueError("scGPT label_map.json keys must be contiguous from zero")

    with open(gene_file, "r") as handle:
        genes = json.load(handle)
    if not isinstance(genes, list) or not all(isinstance(g, str) for g in genes):
        raise ValueError(f"Invalid scGPT gene list: {gene_file}")

    model, vocab, pad_value = _load_pretrained(
        pretrained_dir, len(id2type), device,
    )
    if identity.get("pretrained_load_audit") != dict(
        model._pretrained_load_audit
    ):
        raise RuntimeError(
            "scGPT completed model predates the strict pretrained-weight audit"
        )
    if identity.get("attention_backend") != _attention_backend(model):
        raise RuntimeError(
            "scGPT completed model attention backend changed"
        )

    fine_tuned_state = _torch_load(model_file, map_location="cpu")
    model.load_state_dict(fine_tuned_state, strict=True)
    model.to(device)
    model.eval()

    missing_genes = [gene for gene in genes if gene not in vocab]
    if missing_genes:
        raise RuntimeError(
            f"Saved scGPT gene list contains {len(missing_genes)} genes absent "
            "from the configured vocabulary"
        )
    gene_ids = np.asarray(vocab(genes), dtype=int)
    print(f"[scGPT] loaded completed model from {save_dir}", flush=True)
    return {
        "model": model,
        "vocab": vocab,
        "gene_ids": gene_ids,
        "id2type": id2type,
        "num_types": len(id2type),
        "pad_value": pad_value,
        "pretrained_dir": str(pretrained_dir),
        "save_dir": str(save_dir),
    }


def _load_pretrained(model_dir, num_types, device):
    """Build TransformerModel and load pretrained weights from model_dir."""
    model_dir = Path(model_dir)
    vocab_file = model_dir / "vocab.json"
    config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"

    vocab = GeneVocab.from_file(vocab_file)
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)

    with open(config_file, "r") as f:
        model_configs = json.load(f)

    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs.get("n_layers_cls", 3)

    pad_value = -2

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=embsize,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=num_types,
        vocab=vocab,
        dropout=DEFAULT_DROPOUT,
        pad_value=pad_value,
        ecs_threshold=0.0,
        # Match the official annotation tutorial. scGPT 0.2.4 resolves this
        # request to the PyTorch encoder itself when FlashMHA is unavailable.
        use_fast_transformer=REQUESTED_USE_FAST_TRANSFORMER,
        fast_transformer_backend=REQUESTED_FAST_TRANSFORMER_BACKEND,
    )
    model._requested_use_fast_transformer = REQUESTED_USE_FAST_TRANSFORMER
    model._requested_fast_transformer_backend = (
        REQUESTED_FAST_TRANSFORMER_BACKEND
    )

    pretrained_dict = _torch_load(model_file, map_location="cpu")
    model = _load_pretrained_state_strict(model, pretrained_dict)
    del pretrained_dict
    model.to(device)
    print(
        "[scGPT] attention backend: "
        + json.dumps(_attention_backend(model), sort_keys=True),
        flush=True,
    )

    return model, vocab, pad_value


# ============================================================
# Checkpoint Identity & Resume
# ============================================================

def _training_source_identity(path):
    """Lightweight source metadata for safe interrupted-run resume."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _device_identity(device):
    identity = {"type": device.type}
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        identity.update({
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
        })
    return identity


def _training_identity(
    adata, celltype_ids, id2type, label_col, pretrained_dir,
    model, seed, args, device,
):
    """Describe every input that must match for an optimization resume."""
    label_ids, label_sizes = np.unique(celltype_ids, return_counts=True)
    label_counts = {
        str(id2type[int(label_id)]): int(label_size)
        for label_id, label_size in zip(label_ids, label_sizes)
    }
    return {
        "schema": TRAINING_STATE_SCHEMA,
        "seed": int(seed),
        "label_col": str(label_col),
        "labels": [str(id2type[index]) for index in range(len(id2type))],
        "data": {
            "source": _training_source_identity(args.train_path),
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "first_obs_name": (
                str(adata.obs_names[0]) if adata.n_obs else None
            ),
            "last_obs_name": (
                str(adata.obs_names[-1]) if adata.n_obs else None
            ),
            "label_counts": label_counts,
        },
        "hyperparameters": _training_hyperparameters(args),
        "pretrained": _pretrained_identity(pretrained_dir),
        "pretrained_load_audit": dict(model._pretrained_load_audit),
        "attention_backend": _attention_backend(model),
        "runtime": {
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "device": _device_identity(device),
        },
    }


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("scGPT checkpoint requires CUDA RNG state")
        if len(cuda_state) != torch.cuda.device_count():
            raise RuntimeError(
                "scGPT checkpoint CUDA device count differs from this run"
            )
        torch.cuda.set_rng_state_all(cuda_state)


def _training_state_path(save_dir):
    return Path(save_dir) / TRAINING_STATE_FILE


def _discard_training_state(save_dir):
    path = _training_state_path(save_dir)
    path.unlink(missing_ok=True)
    path.with_name(f".{path.name}.tmp").unlink(missing_ok=True)


def _save_training_state(
    save_dir, identity, epoch, model, optimizer, scheduler, scaler,
    best_state, best_val_loss,
):
    payload = {
        "schema": TRAINING_STATE_SCHEMA,
        "identity": identity,
        "completed_epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_state": best_state,
        "best_val_loss": float(best_val_loss),
        "rng_state": _capture_rng_state(),
    }
    _atomic_torch_save(payload, _training_state_path(save_dir))


def _load_training_state(
    save_dir, identity, model, optimizer, scheduler, scaler,
):
    """Restore a complete epoch, returning None for stale/incomplete state."""
    path = _training_state_path(save_dir)
    path.with_name(f".{path.name}.tmp").unlink(missing_ok=True)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        checkpoint = _torch_load(path, map_location="cpu")
    except Exception as error:
        warnings.warn(f"Ignoring unreadable scGPT training state: {error}")
        path.unlink(missing_ok=True)
        return None

    required = {
        "schema", "identity", "completed_epoch", "model_state",
        "optimizer_state", "scheduler_state", "scaler_state", "best_state",
        "best_val_loss", "rng_state",
    }
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        warnings.warn("Ignoring incomplete scGPT training state")
        path.unlink(missing_ok=True)
        return None
    if (
        checkpoint["schema"] != TRAINING_STATE_SCHEMA
        or checkpoint["identity"] != identity
    ):
        print(
            "[scGPT] partial checkpoint identity changed; restarting training",
            flush=True,
        )
        path.unlink(missing_ok=True)
        return None

    completed_epoch = int(checkpoint["completed_epoch"])
    if completed_epoch < 1:
        warnings.warn("Ignoring scGPT checkpoint with invalid epoch")
        path.unlink(missing_ok=True)
        return None

    try:
        model.load_state_dict(checkpoint.pop("model_state"), strict=True)
        optimizer.load_state_dict(checkpoint.pop("optimizer_state"))
        scheduler.load_state_dict(checkpoint.pop("scheduler_state"))
        scaler.load_state_dict(checkpoint.pop("scaler_state"))
    except Exception as error:
        raise RuntimeError(
            "scGPT checkpoint matched this run but its training state could "
            "not be restored"
        ) from error

    best_state = checkpoint.pop("best_state")
    result = {
        "start_epoch": completed_epoch + 1,
        "best_state": best_state,
        "best_val_loss": float(checkpoint["best_val_loss"]),
    }
    rng_state = checkpoint.pop("rng_state")
    del checkpoint
    _restore_rng_state(rng_state)
    print(
        f"[scGPT] resuming after completed epoch {completed_epoch}",
        flush=True,
    )
    return result


# ============================================================
# Training Loop
# ============================================================

def _train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion_cls,
                     vocab, pad_value, device, use_amp):
    model.train()
    total_loss, total_error, total_num = 0.0, 0.0, 0

    for batch_data in loader:
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)
        src_key_padding_mask = input_gene_ids.eq(vocab[PAD_TOKEN])

        with torch.cuda.amp.autocast(enabled=use_amp):
            output_dict = model(
                input_gene_ids, input_values,
                src_key_padding_mask=src_key_padding_mask,
                CLS=True,
            )
            loss = criterion_cls(output_dict["cls_output"], celltype_labels)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                       error_if_nonfinite=False if scaler.is_enabled() else True)
        scaler.step(optimizer)
        scaler.update()

        bs = input_gene_ids.size(0)
        total_loss += loss.item() * bs
        total_error += (1 - (output_dict["cls_output"].argmax(1) == celltype_labels
                             ).sum().item() / bs) * bs
        total_num += bs

    scheduler.step()
    return total_loss / total_num, total_error / total_num


def _evaluate(model, loader, criterion_cls, vocab, device, use_amp):
    model.eval()
    total_loss, total_error, total_num = 0.0, 0.0, 0

    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[PAD_TOKEN])

            with torch.cuda.amp.autocast(enabled=use_amp):
                output_dict = model(
                    input_gene_ids, input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=True,
                )
                loss = criterion_cls(output_dict["cls_output"], celltype_labels)

            bs = input_gene_ids.size(0)
            total_loss += loss.item() * bs
            total_error += (1 - (output_dict["cls_output"].argmax(1) == celltype_labels
                                 ).sum().item() / bs) * bs
            total_num += bs

    return total_loss / total_num, total_error / total_num


def _predict_labels(model, loader, vocab, device, use_amp):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[PAD_TOKEN])

            with torch.cuda.amp.autocast(enabled=use_amp):
                output_dict = model(
                    input_gene_ids, input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=True,
                )
            all_preds.append(output_dict["cls_output"].argmax(1).cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def _finetune(adata, label_col, pretrained_dir, save_dir, seed, args, device):
    """Full fine-tuning pipeline: preprocess, tokenize, train, return handle."""
    set_seed(seed)
    ensure_dir(save_dir)

    # Build label mapping
    celltypes = adata.obs[label_col].astype("category")
    id2type = dict(enumerate(celltypes.cat.categories))
    num_types = len(id2type)
    celltype_ids = celltypes.cat.codes.values.astype(int)
    adata.obs["celltype_id"] = celltype_ids

    # Load pretrained vocab & model
    model, vocab, pad_value = _load_pretrained(
        pretrained_dir, num_types, device,
    )

    # Copy vocab to save_dir
    shutil.copy(Path(pretrained_dir) / "vocab.json", save_dir / "vocab.json")

    # Preprocess
    adata = _preprocess_adata(adata, vocab)
    genes = adata.var["gene_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
    training_identity = _training_identity(
        adata, celltype_ids, id2type, label_col, pretrained_dir,
        model, seed, args, device,
    )

    # Split row indices first so sparse binned data never needs a full dense
    # n_cells x n_genes materialization. Do not pass random_state here: the
    # official v0.2.4 tutorial continues the global NumPy RNG sequence seeded
    # by set_seed(seed) above and advanced by preprocessing.
    layer = adata.layers["X_binned"]
    row_indices = np.arange(adata.n_obs)
    (train_indices, valid_indices,
     train_ct, valid_ct) = train_test_split(
        row_indices, celltype_ids,
        test_size=0.1, shuffle=True,
    )
    train_counts = layer[train_indices]
    valid_counts = layer[valid_indices]

    # Tokenize
    tokenized_train = _tokenize_matrix(
        train_counts, gene_ids, vocab, pad_value
    )
    tokenized_valid = _tokenize_matrix(
        valid_counts, gene_ids, vocab, pad_value
    )

    # Build data tensors
    def _make_pt(tok, ct):
        return {
            "gene_ids": tok["genes"],
            # Preserve the tutorial's RNG state for later query tokenization.
            "values": random_mask_value(
                tok["values"], mask_ratio=0.0, pad_value=pad_value,
            ),
            "celltype_labels": torch.from_numpy(ct).long(),
        }

    train_pt = _make_pt(tokenized_train, train_ct)
    valid_pt = _make_pt(tokenized_valid, valid_ct)

    batch_size = args.batch_size
    train_loader = _prepare_dataloader(
        train_pt,
        batch_size,
        shuffle=False,
        num_workers=args.n_jobs,
    )
    valid_loader = _prepare_dataloader(
        valid_pt,
        batch_size,
        shuffle=False,
        num_workers=args.n_jobs,
    )

    # Optimizer / scheduler
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 1, gamma=args.schedule_ratio,
    )
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Training. A killed process restarts only the interrupted epoch; all
    # completed epochs retain their exact optimizer, scheduler and RNG state.
    best_val_loss = float("inf")
    best_state = None
    start_epoch = 1
    resume = _load_training_state(
        save_dir, training_identity, model, optimizer, scheduler, scaler,
    )
    if resume is not None:
        start_epoch = resume["start_epoch"]
        best_state = resume["best_state"]
        best_val_loss = resume["best_val_loss"]

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_err = _train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion_cls, vocab, pad_value, device, use_amp,
        )
        val_loss, val_err = _evaluate(
            model, valid_loader, criterion_cls, vocab, device, use_amp,
        )
        elapsed = time.time() - t0
        print(f"  epoch {epoch:3d} | {elapsed:5.1f}s | "
              f"train loss {train_loss:.4f} err {train_err:.4f} | "
              f"valid loss {val_loss:.4f} err {val_err:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        _save_training_state(
            save_dir, training_identity, epoch, model, optimizer, scheduler,
            scaler, best_state, best_val_loss,
        )

    # Publish all inference artifacts atomically, then remove partial state.
    if best_state is None:
        raise RuntimeError("scGPT training did not produce a valid checkpoint")
    _atomic_torch_save(best_state, save_dir / "best_model.pt")
    model.load_state_dict(best_state)
    _atomic_json_dump(
        {str(k): str(v) for k, v in id2type.items()},
        save_dir / "label_map.json",
    )
    _atomic_json_dump(genes, save_dir / "gene_list.json")
    _atomic_json_dump(training_identity, save_dir / "model_identity.json")
    _discard_training_state(save_dir)

    del train_loader, valid_loader, scaler, optimizer, scheduler, best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model,
        "vocab": vocab,
        "gene_ids": gene_ids,
        "id2type": id2type,
        "num_types": num_types,
        "pad_value": pad_value,
        "pretrained_dir": str(pretrained_dir),
        "save_dir": str(save_dir),
    }


# ============================================================
# Train / Predict  (pipeline interface)
# ============================================================

def train_scgpt(train_path, model_dir, label_col, seed, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata = load_adata(train_path)
    resolved = resolve_label_col(adata, label_col)
    print(f"[train] {train_path} | label={resolved} | seed={seed}")

    handle = _finetune(
        adata, resolved, args.scgpt_model_dir,
        ensure_dir(model_dir), seed, args, device,
    )
    return handle


def predict_scgpt(test_path, model_handle, pred_col, seed, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata = load_adata(test_path)

    model = model_handle["model"]
    vocab = model_handle["vocab"]
    id2type = model_handle["id2type"]
    pad_value = model_handle["pad_value"]

    # Preprocess test data with same vocab
    adata_proc = _preprocess_adata(adata.copy(), vocab)
    genes = adata_proc.var["gene_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)

    tokenized = _tokenize(adata_proc, gene_ids, vocab, pad_value)
    test_pt = {
        "gene_ids": tokenized["genes"],
        "values": random_mask_value(
            tokenized["values"], mask_ratio=0.0,
            pad_value=pad_value,
        ),
        "target_values": tokenized["values"],
        "celltype_labels": torch.zeros(adata_proc.n_obs, dtype=torch.long),
    }

    loader = _prepare_dataloader(
        test_pt,
        args.batch_size,
        shuffle=False,
        num_workers=args.n_jobs,
    )
    model.to(device)
    use_amp = True
    preds_ids = _predict_labels(model, loader, vocab, device, use_amp)
    preds = pd.Series([id2type[int(p)] for p in preds_ids], index=adata.obs_names)

    out = adata.copy()
    out.obs[pred_col] = preds.values
    return out


# ============================================================
# Subtype (fine-tune per broad type)
# ============================================================

def train_subtype_scgpt(subtype_train_path, model_dir, seed, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata = load_adata(subtype_train_path)
    subtype_col = "Celltype_subtraining"
    broad_col = ("Celltype_training"
                 if "Celltype_training" in adata.obs.columns else None)
    if broad_col is None:
        adata.obs["_derived_broad"] = (
            adata.obs[subtype_col].astype(str).map(get_broad_type)
        )
        broad_col = "_derived_broad"

    ensure_dir(model_dir)

    # Broad-type model
    print(f"[subtype-train] broad model ...")
    broad_handle = _finetune(
        adata.copy(), broad_col, args.scgpt_model_dir,
        ensure_dir(model_dir / "broad"), seed, args, device,
    )

    # Per-broad-type subtype models
    subtype_handles = {}
    for bt in BROAD_TYPES_FOR_SUBTYPE:
        mask = adata.obs[broad_col].astype(str) == bt
        subset = adata[mask].copy()
        if subset.n_obs == 0 or subset.obs[subtype_col].nunique() < 2:
            continue
        safe = bt.replace(" ", "_").replace("+", "plus")
        print(f"[subtype-train] {bt} ...")
        sub_handle = _finetune(
            subset, subtype_col, args.scgpt_model_dir,
            ensure_dir(model_dir / f"subtype_{safe}"), seed, args, device,
        )
        subtype_handles[bt] = sub_handle
        print(f"[subtype-train] {bt}: done")

    return broad_handle, subtype_handles


def subtype_predict_scgpt(predict_fn, test_path, broad_handle,
                          subtype_handles, output_dir, tool_name,
                          pred_col, seed, args):
    """Predict broad then refine per-type with subtype models."""
    prefix = pred_col.replace("_pred", "")
    bp, sp = f"{prefix}_broad_pred", f"{prefix}_subtype_pred"

    # Broad prediction
    adata = predict_fn(test_path, broad_handle, bp, seed, args)
    out = adata.copy()
    out.obs[sp] = out.obs[bp].astype(str)
    out.obs[pred_col] = out.obs[bp].astype(str)

    # Per-broad-type subtype refinement
    ensure_dir(output_dir)
    for bt, sub_h in subtype_handles.items():
        mask = out.obs[bp].astype(str) == bt
        if mask.sum() == 0:
            continue
        subset = out[mask].copy()
        safe = bt.replace(" ", "_").replace("+", "plus")
        tmp = output_dir / f"_tmp_{safe}.h5ad"
        subset.write_h5ad(tmp)

        a_sub = predict_fn(tmp, sub_h, sp, seed, args)
        preds = a_sub.obs[sp].astype(str)
        broad_check = preds.map(get_broad_type)
        accepted = preds.where(broad_check == bt, bt)

        out.obs.loc[subset.obs_names, sp] = preds.values
        out.obs.loc[subset.obs_names, pred_col] = accepted.values
        tmp.unlink(missing_ok=True)

    save_subtype_results(out, pred_col, bp, sp, test_path, output_dir, tool_name)
