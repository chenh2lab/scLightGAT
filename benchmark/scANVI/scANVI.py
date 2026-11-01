#!/usr/bin/env python3
"""scANVI reference mapping with the official scvi-tools workflow."""

import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import scvi
from anndata import AnnData

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.pipeline import (  # noqa: E402
    SCORE_CLASSES_KEY, SCORE_KEY, create_argument_parser, run_annotation_pipeline,
)


BATCH_KEY = "_scanvi_batch"
UNLABELED_CATEGORY = "__scanvi_unlabeled__"


def _configure_runtime(seed: int, args) -> None:
    scvi.settings.seed = seed
    scvi.settings.num_threads = args.n_jobs


def _set_batch(adata: AnnData, source_key: str | None, default: str) -> None:
    adata.obs[BATCH_KEY] = (
        default if source_key is None else adata.obs[source_key].astype(str)
    )


def _training_kwargs(args, max_epochs: int | None) -> dict:
    return {
        "accelerator": args.accelerator,
        "devices": args.devices,
        "batch_size": args.batch_size,
        **({"max_epochs": max_epochs} if max_epochs is not None else {}),
    }


def train_scanvi(adata: AnnData, seed: int, run_out_path: Path, args) -> Path:
    """Train and save one SCVI-to-SCANVI reference model."""
    _configure_runtime(seed, args)
    _set_batch(adata, args.train_batch, "reference")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=args.n_top_genes,
        batch_key=BATCH_KEY,
        subset=True,
    )

    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=BATCH_KEY)
    scvi_model = scvi.model.SCVI(
        adata,
        use_layer_norm="both",
        use_batch_norm="none",
        encode_covariates=True,
        dropout_rate=0.2,
        n_layers=2,
    )
    scvi_model.train(**_training_kwargs(args, args.scvi_max_epochs))

    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        unlabeled_category=UNLABELED_CATEGORY,
        labels_key=args.train_celltype,
    )
    scanvi_model.train(
        n_samples_per_label=args.n_samples_per_label,
        **_training_kwargs(args, args.scanvi_max_epochs),
    )

    reference_path = Path(run_out_path) / "_model"
    scanvi_model.save(str(reference_path), overwrite=True)
    return reference_path


def predict_scanvi(
    adata: AnnData, reference_path: Path, pred_key: str, seed: int,
    run_out_path: Path, args,
) -> AnnData:
    """Map one query from the unchanged saved reference model."""
    del run_out_path
    _configure_runtime(seed, args)
    query = AnnData(
        X=adata.X, obs=adata.obs.copy(), var=adata.var.copy(),
        layers={"counts": adata.layers["counts"]},
    )
    _set_batch(query, args.test_batch, "query")

    reference_path = str(reference_path)
    scvi.model.SCANVI.prepare_query_anndata(query, reference_path)
    query_model = scvi.model.SCANVI.load_query_data(
        query, reference_path, accelerator=args.accelerator, device="auto",
    )
    query_model.train(
        check_val_every_n_epoch=args.query_check_val_every_n_epoch,
        plan_kwargs={"weight_decay": 0.0},
        **_training_kwargs(args, args.query_max_epochs),
    )

    probabilities = query_model.predict(query, soft=True, batch_size=args.batch_size)
    adata.obs[pred_key] = probabilities.idxmax(axis=1).astype(str).to_numpy()
    adata.obsm[SCORE_KEY] = probabilities.to_numpy(dtype=float)
    adata.uns[SCORE_CLASSES_KEY] = np.asarray(probabilities.columns, dtype=str)
    adata.uns["annotation_score_type"] = "probabilities"
    return adata


if __name__ == "__main__":
    parser = create_argument_parser("scANVI", include_n_jobs=True)
    parser.set_defaults(
        n_top_genes=2_000,
        batch_size=128,
        scvi_max_epochs=None,
        scanvi_max_epochs=20,
        n_samples_per_label=100,
        query_max_epochs=100,
        query_check_val_every_n_epoch=10,
        accelerator="auto",
        devices="auto",
    )
    arguments = parser.parse_args()

    run_annotation_pipeline(
        "scANVI", "scanvi_prediction", train_scanvi, predict_scanvi, arguments,
        expression_layers=("counts",),
        share_train_expression=True,
        keep_var_metadata=False,
    )
