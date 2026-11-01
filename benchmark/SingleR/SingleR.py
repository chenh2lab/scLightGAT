#!/usr/bin/env python3
"""SingleR reference-based cell type annotation benchmark."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.pipeline import (  # noqa: E402
    create_argument_parser,
    predict_r_reference,
    run_annotation_pipeline,
    train_r_reference,
)

R_SCRIPT = Path(__file__).with_suffix(".R").resolve()


def _check_r_script() -> None:
    if not R_SCRIPT.is_file():
        raise FileNotFoundError(
            f"SingleR R script was not found next to this file: {R_SCRIPT}"
        )


def _worker_args(args: Any) -> list[str]:
    """Pass the benchmark-wide CPU allowance to BiocParallel/SingleR."""
    return ["--n_jobs", str(args.n_jobs)]


def train_singler(adata: Any, seed: int, run_out_path: str, args: Any):
    """Save the normalized labelled reference used by SingleR."""
    _check_r_script()
    return train_r_reference(
        adata,
        seed,
        run_out_path,
        args,
        R_SCRIPT,
        extra_args=_worker_args(args),
    )


def predict_singler(
    adata: Any,
    model_path: str,
    pred_key: str,
    seed: int,
    run_out_path: str,
    args: Any,
):
    """Annotate query cells with the saved SingleR reference."""
    _check_r_script()
    return predict_r_reference(
        adata,
        model_path,
        pred_key,
        seed,
        run_out_path,
        args,
        R_SCRIPT,
        score_type="continuous_scores",
        extra_args=_worker_args(args),
    )


if __name__ == "__main__":
    parser = create_argument_parser("SingleR", include_n_jobs=True)
    arguments = parser.parse_args()

    run_annotation_pipeline(
        "SingleR",
        "singler_prediction",
        train_singler,
        predict_singler,
        arguments,
        expression_layers=(),
        share_train_expression=True,
        keep_var_metadata=False,
    )
