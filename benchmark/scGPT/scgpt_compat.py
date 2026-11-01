"""Compatibility names used by the preserved scGPT model implementation."""

from pathlib import Path

from utils.dataset import load_dataset
from utils.pipeline import (
    create_argument_parser,
    run_annotation_pipeline,
    seed_python_numpy,
)

BROAD_TYPES_FOR_SUBTYPE = [
    "CD4+T cells", "CD8+T cells", "B cells", "Plasma cells", "DC"
]


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_adata(path):
    return load_dataset(path)


def resolve_label_col(adata, preferred=None):
    if preferred and preferred in adata.obs:
        return preferred
    raise KeyError(f"Label column '{preferred}' not found")


def save_subtype_results(*args, **kwargs):
    raise NotImplementedError("The generic train/test benchmark does not use subtype mode")
