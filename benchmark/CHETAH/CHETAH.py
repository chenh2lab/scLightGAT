"""CHETAH reference-based cell type annotation benchmark."""

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


def train_chetah(adata, seed, run_out_path, args):
    return train_r_reference(adata, seed, run_out_path, args, R_SCRIPT)


def predict_chetah(adata, model_path, pred_key, seed, run_out_path, args):
    # CHETAH's standard output is a discrete hierarchy assignment. The common
    # evaluator therefore records AUC as hard_predictions for this method.
    return predict_r_reference(
        adata, model_path, pred_key, seed, run_out_path, args, R_SCRIPT
    )


if __name__ == "__main__":
    parser = create_argument_parser("CHETAH", include_n_jobs=True)
    arguments = parser.parse_args()
    run_annotation_pipeline(
        "CHETAH", "chetah_prediction", train_chetah, predict_chetah, arguments,
        expression_layers=(),
        share_train_expression=True,
        keep_var_metadata=False,
    )
