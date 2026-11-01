"""Fit-once, predict-many facade for the scLightGAT pipeline."""

import logging
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import torch

from .GAT import GATModel
from .LightGBM import LightGBMModel
from .data import (
    align_adata_to_genes,
    compute_reference_gene_means,
    prepare_lightgbm_data,
    select_dge_genes,
)
from .graph import build_gat_data, build_neighbor_graph
from .hparam import (
    BATCH_MODE,
    DGE_FEATURES_PER_TYPE,
    GAT_CLASS_WEIGHT_EXPONENT,
    GAT_CLASS_WEIGHT_MAX,
    GAT_DROPOUT,
    GAT_EPOCHS,
    GAT_HEADS,
    GAT_HIDDEN_CHANNELS,
    GAT_LEARNING_RATE,
    GAT_WEIGHT_DECAY,
    GRAPH_METRIC,
    GRAPH_N_NEIGHBORS,
    LIGHTGBM_BAGGING_FRACTION,
    LIGHTGBM_BAGGING_FREQ,
    LIGHTGBM_CLASS_WEIGHT,
    LIGHTGBM_FEATURE_FRACTION,
    LIGHTGBM_LEARNING_RATE,
    LIGHTGBM_MAX_DEPTH,
    LIGHTGBM_MIN_DATA_IN_LEAF,
    LIGHTGBM_N_ESTIMATORS,
    LIGHTGBM_NUM_LEAVES,
    LIGHTGBM_REG_ALPHA,
    LIGHTGBM_REG_LAMBDA,
)
from .random import seed_everything


logger = logging.getLogger(__name__)


def _default_hparams():
    return {
        "batch_mode": BATCH_MODE,
        "features": {
            "dge_features_per_type": DGE_FEATURES_PER_TYPE,
        },
        "gat": {
            "class_weight_exponent": GAT_CLASS_WEIGHT_EXPONENT,
            "class_weight_max": GAT_CLASS_WEIGHT_MAX,
            "dropout": GAT_DROPOUT,
            "epochs": GAT_EPOCHS,
            "heads": GAT_HEADS,
            "hidden_channels": GAT_HIDDEN_CHANNELS,
            "learning_rate": GAT_LEARNING_RATE,
            "weight_decay": GAT_WEIGHT_DECAY,
        },
        "graph": {
            "metric": GRAPH_METRIC,
            "n_neighbors": GRAPH_N_NEIGHBORS,
        },
        "lightgbm": {
            "bagging_fraction": LIGHTGBM_BAGGING_FRACTION,
            "bagging_freq": LIGHTGBM_BAGGING_FREQ,
            "class_weight": LIGHTGBM_CLASS_WEIGHT,
            "feature_fraction": LIGHTGBM_FEATURE_FRACTION,
            "learning_rate": LIGHTGBM_LEARNING_RATE,
            "max_depth": LIGHTGBM_MAX_DEPTH,
            "min_data_in_leaf": LIGHTGBM_MIN_DATA_IN_LEAF,
            "n_estimators": LIGHTGBM_N_ESTIMATORS,
            "num_leaves": LIGHTGBM_NUM_LEAVES,
            "reg_alpha": LIGHTGBM_REG_ALPHA,
            "reg_lambda": LIGHTGBM_REG_LAMBDA,
        },
    }


def _merge_hparams(overrides=None):
    merged = _default_hparams()
    if overrides is None:
        return merged

    def merge(target, patch):
        for key, value in patch.items():
            if key not in target:
                continue
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = deepcopy(value)

    merge(merged, overrides)
    return merged


class scLightGAT:
    """Train one reference model and apply it to independent queries."""

    def __init__(self, seed=0, hparams=None):
        selected = _merge_hparams(hparams)

        self.hparams = selected
        self.seed = int(seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dge_genes = []
        self.hvg_genes = []
        self.dge_reference_means = {}
        self.hvg_reference_means = {}
        self.hvg_method = None
        self.label_encoder = None
        self.gat_in_channels = None
        self.gat = None
        self.lightgbm = LightGBMModel(
            config=selected["lightgbm"],
            random_state=self.seed,
        )
        self.fitted = False
        seed_everything(self.seed)

    def fit_reference(
        self,
        adata,
        label_key="Celltype_training",
        batch_key=None,
    ):
        """Fit LightGBM and GAT using labelled reference cells only."""
        seed_everything(self.seed)
        if self.hparams["batch_mode"] == "none":
            batch_key = None
        if label_key not in adata.obs:
            raise KeyError(f"Training label column '{label_key}' not found")

        valid = adata.obs[label_key].notna()
        valid &= adata.obs[label_key].astype(str).str.strip().ne("")
        reference = adata if bool(valid.all()) else adata[valid].copy()
        if reference.obs[label_key].nunique() < 2:
            raise ValueError("Training requires at least two cell types")
        reference.raw = None
        for layer in list(reference.layers):
            if layer not in {"counts", "log_transformed"}:
                del reference.layers[layer]

        self.dge_genes = select_dge_genes(
            reference,
            celltype_column=label_key,
            top_features=self.hparams["features"]["dge_features_per_type"],
        )
        self.hvg_genes = []
        self.hvg_method = "disabled"

        self.dge_reference_means = compute_reference_gene_means(
            reference, self.dge_genes
        )
        dge_reference = align_adata_to_genes(
            reference,
            self.dge_genes,
            reference_means=self.dge_reference_means,
        )
        dge_features, encoded_labels, self.label_encoder = (
            prepare_lightgbm_data(
                dge_reference, celltype_column=label_key
            )
        )
        encoded_labels = np.asarray(encoded_labels)

        self.hvg_reference_means = {}
        lightgbm_features = dge_features

        if "counts" in dge_reference.layers:
            del dge_reference.layers["counts"]
        oof_probabilities = self.lightgbm.fit(
            lightgbm_features, encoded_labels
        )

        build_neighbor_graph(
            dge_reference,
            batch_key=batch_key,
            seed=self.seed,
            **self.hparams["graph"],
        )
        gat_data = build_gat_data(dge_reference, oof_probabilities)
        gat_data.train_mask = torch.ones(
            gat_data.num_nodes, dtype=torch.bool
        )
        self.gat_in_channels = int(gat_data.x.shape[1])
        self.gat = GATModel(
            self.gat_in_channels,
            len(self.label_encoder.classes_),
            config=self.hparams["gat"],
        )
        self.gat.fit(gat_data, encoded_labels, seed=self.seed)

        self.fitted = True
        return self

    def predict_query(
        self,
        adata,
        batch_key=None,
        pred_key="scLightGAT_pred",
    ):
        """Predict without reading query cell-type labels."""
        if not self.fitted or self.label_encoder is None:
            raise ValueError("The reference model has not been fitted")
        seed_everything(self.seed)
        if self.hparams["batch_mode"] == "none":
            batch_key = None

        dge_query = align_adata_to_genes(
            adata,
            self.dge_genes,
            reference_means=self.dge_reference_means,
        )
        missing_dge = dge_query.var[
            "scLightGAT_missing_in_source"
        ].to_numpy(dtype=bool)
        if "counts" in dge_query.layers:
            del dge_query.layers["counts"]
        if "log_transformed" not in dge_query.layers:
            raise KeyError("Query is missing required layer 'log_transformed'")
        dge_features = dge_query.to_df(
            layer="log_transformed"
        ).reindex(columns=self.dge_genes)
        if not np.isfinite(dge_features.to_numpy()).all():
            raise ValueError("Aligned DGE query contains non-finite values")

        missing_hvg = np.zeros(0, dtype=bool)
        lightgbm_features = dge_features

        dge_coverage = float(1.0 - missing_dge.mean())
        hvg_coverage = (
            float(1.0 - missing_hvg.mean())
            if missing_hvg.size
            else 1.0
        )
        logger.info(
            "Query feature coverage: DGE %.2f%% (%d/%d), HVG %.2f%% (%d/%d)",
            100.0 * dge_coverage,
            int((~missing_dge).sum()),
            len(self.dge_genes),
            100.0 * hvg_coverage,
            int((~missing_hvg).sum()),
            len(self.hvg_genes),
        )

        lightgbm_encoded = self.lightgbm.predict(
            lightgbm_features
        ).astype(int)
        lightgbm_probabilities = self.lightgbm.predict_proba(
            lightgbm_features
        ).astype(np.float32)
        raw_predictions = self.label_encoder.inverse_transform(
            lightgbm_encoded
        )

        build_neighbor_graph(
            dge_query,
            batch_key=batch_key,
            seed=self.seed,
            **self.hparams["graph"],
        )
        gat_data = build_gat_data(dge_query, lightgbm_probabilities)
        probabilities = self.gat.predict_proba(gat_data)
        encoded = probabilities.argmax(axis=1)
        score_type = "gat_probabilities"

        result = adata
        result.obs["scLightGAT_rawpred"] = raw_predictions
        result.obs[pred_key] = self.label_encoder.inverse_transform(encoded)
        result.obsm["annotation_scores"] = probabilities.astype(np.float32)
        result.uns["annotation_score_classes"] = np.asarray(
            self.label_encoder.classes_, dtype=str
        )
        result.uns["annotation_score_type"] = score_type
        result.uns["scLightGAT_seed"] = self.seed
        result.uns["scLightGAT_hvg_method"] = self.hvg_method
        result.uns["scLightGAT_missing_gene_imputation"] = (
            "train_reference_gene_mean_during_schema_alignment_per_matrix_scale"
        )
        result.uns["scLightGAT_dge_gene_coverage"] = dge_coverage
        result.uns["scLightGAT_missing_dge_genes"] = np.asarray(
            np.asarray(self.dge_genes, dtype=str)[missing_dge],
            dtype=str,
        )
        return result

    def save(self, model_dir):
        """Persist the fitted reference model."""
        if not self.fitted:
            raise ValueError("Cannot save an unfitted model")
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "seed": self.seed,
            "hparams": self.hparams,
            "dge_genes": self.dge_genes,
            "hvg_genes": self.hvg_genes,
            "dge_reference_means": self.dge_reference_means,
            "hvg_reference_means": self.hvg_reference_means,
            "hvg_method": self.hvg_method,
            "label_encoder": self.label_encoder,
            "gat_in_channels": self.gat_in_channels,
        }
        joblib.dump(metadata, model_dir / "metadata.pkl")
        self.lightgbm.save(model_dir / "lightgbm_model.pkl")
        torch.save(
            {
                "state_dict": self.gat.state_dict(),
                "in_channels": self.gat_in_channels,
                "out_channels": self.gat.out_channels,
            },
            model_dir / "gat_model.pt",
        )

    @classmethod
    def load(cls, model_dir):
        """Restore a model created by :meth:`save`."""
        model_dir = Path(model_dir)
        metadata = joblib.load(model_dir / "metadata.pkl")
        model = cls(
            seed=metadata["seed"],
            hparams=metadata["hparams"],
        )
        model.dge_genes = list(metadata["dge_genes"])
        model.hvg_genes = list(metadata["hvg_genes"])
        model.dge_reference_means = cls._validate_reference_means(
            metadata["dge_reference_means"], model.dge_genes, "DGE"
        )
        model.hvg_reference_means = cls._validate_reference_means(
            metadata["hvg_reference_means"], model.hvg_genes, "HVG"
        )
        model.hvg_method = metadata.get("hvg_method")
        model.label_encoder = metadata["label_encoder"]
        model.gat_in_channels = metadata.get("gat_in_channels")
        model.lightgbm.load(model_dir / "lightgbm_model.pkl")

        checkpoint = torch.load(
            model_dir / "gat_model.pt",
            map_location=model.device,
            weights_only=True,
        )
        model.gat = GATModel(
            checkpoint["in_channels"],
            checkpoint["out_channels"],
            config=model.hparams["gat"],
        ).to(model.device)
        model.gat.load_state_dict(checkpoint["state_dict"])
        model.gat.eval()
        model.fitted = True
        return model

    @classmethod
    def is_complete(cls, model_dir):
        model_dir = Path(model_dir)
        metadata_path = model_dir / "metadata.pkl"
        if not metadata_path.is_file():
            return False
        if not (model_dir / "lightgbm_model.pkl").is_file():
            return False
        try:
            metadata = joblib.load(metadata_path)
        except Exception:
            return False
        hparams = metadata.get("hparams")
        if not isinstance(hparams, dict):
            return False
        if not (model_dir / "gat_model.pt").is_file():
            return False
        return True

    @staticmethod
    def _validate_reference_means(reference_means, genes, schema_name):
        if not hasattr(reference_means, "items"):
            raise ValueError(f"Saved {schema_name} reference means are invalid")
        restored = {}
        for matrix_name, values in reference_means.items():
            array = np.asarray(values, dtype=np.float32).reshape(-1)
            if array.shape != (len(genes),):
                raise ValueError(
                    f"Saved {schema_name} {matrix_name} means have shape "
                    f"{array.shape}; expected ({len(genes)},)"
                )
            if not np.isfinite(array).all():
                raise ValueError(
                    f"Saved {schema_name} {matrix_name} means are non-finite"
                )
            restored[str(matrix_name)] = array
        for required in ("X", "log_transformed"):
            if genes and required not in restored:
                raise ValueError(
                    f"Saved {schema_name} means are missing '{required}'"
                )
        return restored
