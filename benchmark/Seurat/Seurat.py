"""Seurat anchor-based label-transfer annotation benchmark."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.pipeline import (
    create_argument_parser,
    predict_r_reference,
    run_annotation_pipeline,
    train_r_reference,
)

R_SCRIPT = Path(__file__).with_suffix(".R")


def train_seurat(adata, seed, run_out_path, args):
    layer = "counts" if "counts" in adata.layers else None
    return train_r_reference(
        adata, seed, run_out_path, args, R_SCRIPT, layer=layer
    )


def predict_seurat(adata, model_path, pred_key, seed, run_out_path, args):
    layer = "counts" if "counts" in adata.layers else None
    return predict_r_reference(
        adata,
        model_path,
        pred_key,
        seed,
        run_out_path,
        args,
        R_SCRIPT,
        score_type="probabilities",
        layer=layer,
    )


if __name__ == "__main__":
    parser = create_argument_parser("Seurat", include_n_jobs=True)
    arguments = parser.parse_args()
    run_annotation_pipeline(
        "Seurat", "seurat_prediction", train_seurat, predict_seurat, arguments,
        expression_layers=("counts",),
        share_train_expression=True,
        keep_var_metadata=False,
    )
