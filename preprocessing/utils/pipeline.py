from __future__ import annotations

import gc
import importlib
import shutil
import traceback
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import pandas as pd

from utils.annotation import annotate_dataset
from utils.checkpoint import Checkpoints, STAGES, code_fingerprint
from utils.config import (
    ALL_LOGICAL_DATASETS,
    CAF_DATASETS,
    EXTRA_DATASETS,
    MAIN_MAJOR_LABELS,
    CAF_TARGET_LABELS,
    OUTPUT_PATHS,
    TEST_DATASETS,
    TRAIN_DATASETS,
)
from utils.io import atomic_write_h5ad, concat_intersection
from utils.donor import DONOR_OBS_COLUMNS, attach_donor_metadata
from utils.processing import (
    ensure_counts,
    normalize_log,
    percentile_qc,
    run_scrublet,
    scdiffusion_qc,
)
from utils.schema import finalize_schema
from utils.subtype import (
    SUBTYPE_OBS_COLUMNS,
    attach_source_subtypes,
    subtype_outputs_complete,
    write_subtype_splits,
)


class ReconstructionPipeline:
    def __init__(self, cache_dir: Path, output: Path):
        self.cache_dir = cache_dir.resolve()
        self.output = output.resolve()
        self.raw = self.cache_dir / "raw"
        self.intermediate = self.cache_dir / "intermediate"
        self.aux = self.cache_dir / "aux"
        self.reports = self.cache_dir / "reports"
        for path in (
            self.raw,
            self.intermediate,
            self.aux,
            self.reports,
            self.output,
        ):
            path.mkdir(parents=True, exist_ok=True)

        project_root = Path(__file__).resolve().parent.parent
        self.checkpoints = Checkpoints(
            self.cache_dir / "checkpoints",
            fingerprint=code_fingerprint(project_root),
        )

    def _artifact(self, dataset: str, stage: str) -> Path:
        return self.intermediate / dataset / f"{stage}.h5ad"

    def _save(self, adata, dataset: str, stage: str, meta=None) -> Path:
        path = self._artifact(dataset, stage)
        atomic_write_h5ad(adata, path)
        self.checkpoints.mark(dataset, stage, path, meta)
        return path

    def _load(self, dataset: str, stage: str):
        checkpoint_path = self.checkpoints.artifact(dataset, stage)
        path = Path(checkpoint_path) if checkpoint_path else self._artifact(dataset, stage)
        return ad.read_h5ad(path)

    def _load_counts_only(self, dataset: str, stage: str):
        """Load one checkpoint matrix without duplicating its count layers."""
        checkpoint_path = self.checkpoints.artifact(dataset, stage)
        path = Path(checkpoint_path) if checkpoint_path else self._artifact(dataset, stage)
        source = ad.read_h5ad(path, backed="r")
        try:
            matrix = source.X
            if hasattr(matrix, "to_memory"):
                matrix = matrix.to_memory()
            else:
                matrix = matrix[...]
            return ad.AnnData(
                X=matrix,
                obs=source.obs.copy(),
                var=source.var.copy(),
                uns=dict(source.uns),
            )
        finally:
            source.file.close()

    def _attach_train_source_annotations(self, train):
        """Restore model labels from each parsed source after integration.

        The integrated count matrix intentionally keeps only genes shared by
        all five inputs.  Source metadata columns are dataset-specific, so an
        inner concat cannot retain them.  Resolve each source with its own
        mapper and join the resulting common annotation schema by cell ID.
        """
        frames = []
        annotation_columns = [
            "Celltype_training",
            "Celltype_provenance",
            "Major_source_label",
            "Major_resolution_source",
            "Celltype_coarse_source",
            "eligible_broad",
            "annotation_status",
            "exclusion_reason",
            *SUBTYPE_OBS_COLUMNS,
            *DONOR_OBS_COLUMNS,
        ]

        for dataset in TRAIN_DATASETS:
            mask = train.obs["Data_source"].astype(str).eq(dataset)
            wanted = pd.Index(
                train.obs.loc[mask, "source_cell_id"].astype(str),
                dtype=str,
            )
            print(
                f"[Integrated_training] resolve {dataset} annotations "
                f"for {len(wanted):,} QC-passed cells",
                flush=True,
            )

            checkpoint_path = self.checkpoints.artifact(dataset, "parsed")
            path = (
                Path(checkpoint_path)
                if checkpoint_path
                else self._artifact(dataset, "parsed")
            )
            source = ad.read_h5ad(path, backed="r")
            try:
                missing = wanted.difference(source.obs_names.astype(str))
                if len(missing):
                    raise ValueError(
                        f"{dataset}: {len(missing):,} integrated source_cell_id "
                        "values are absent from the parsed checkpoint"
                    )
                holder = SimpleNamespace(obs=source.obs.reindex(wanted).copy())
            finally:
                source.file.close()

            annotate_dataset(holder, dataset)
            attach_source_subtypes(holder, dataset)
            attach_donor_metadata(holder, dataset)
            frame = holder.obs[annotation_columns].copy()
            frame.index = pd.Index(
                [f"{dataset}:{value}" for value in frame.index.astype(str)],
                dtype=str,
            )
            frames.append(frame)

        annotations = pd.concat(frames, axis=0)
        missing = train.obs_names.difference(annotations.index)
        if len(missing):
            raise ValueError(
                f"Integrated_training: {len(missing):,} cells have no source "
                "annotation row"
            )
        annotations = annotations.reindex(train.obs_names)
        for column in annotation_columns:
            train.obs[column] = annotations[column]
        return train

    def _integrated_fate_inputs(self):
        """Recover all integrated IDs and saved doublet IDs when resuming."""
        fate_path = self.reports / "Integrated_training_cell_fate.csv"
        if fate_path.exists():
            fate = pd.read_csv(fate_path, dtype=str)
            all_ids = pd.Index(fate["cell_id"].astype(str), dtype=str)
            doublet_ids = pd.Index(
                fate.loc[
                    fate["annotation_status"].eq("predicted_doublet"),
                    "cell_id",
                ].astype(str),
                dtype=str,
            )
            return all_ids, doublet_ids

        all_ids = []
        for dataset in TRAIN_DATASETS:
            checkpoint_path = self.checkpoints.artifact(dataset, "parsed")
            path = (
                Path(checkpoint_path)
                if checkpoint_path
                else self._artifact(dataset, "parsed")
            )
            source = ad.read_h5ad(path, backed="r")
            try:
                all_ids.extend(
                    f"{dataset}:{value}"
                    for value in source.obs_names.astype(str)
                )
            finally:
                source.file.close()
        return pd.Index(all_ids, dtype=str), pd.Index([], dtype=str)

    def _write_cell_fate(
        self,
        dataset: str,
        annotated,
        all_ids=None,
        doublet_ids=None,
    ) -> None:
        if all_ids is None:
            parsed_path = self._artifact(dataset, "parsed")
            parsed = ad.read_h5ad(parsed_path, backed="r")
            all_ids = parsed.obs_names.astype(str).copy()
            parsed.file.close()

        fate = pd.DataFrame({"cell_id": pd.Index(all_ids, dtype=str)})
        fate["dataset"] = dataset
        fate["passed_qc"] = fate["cell_id"].isin(annotated.obs_names)
        audit_columns = [
            "Celltype_coarse_source",
            "Celltype_training",
            "eligible_broad",
            "annotation_status",
            "exclusion_reason",
        ]
        lookup = annotated.obs.reindex(fate["cell_id"])
        for column in audit_columns:
            if column in lookup:
                fate[column] = lookup[column].astype("object").to_numpy()
        failed = ~fate["passed_qc"]
        fate.loc[failed, "annotation_status"] = "failed_qc"
        fate.loc[failed, "exclusion_reason"] = "failed_qc"
        if doublet_ids is not None:
            doublet = fate["cell_id"].isin(doublet_ids)
            fate.loc[doublet, "annotation_status"] = "predicted_doublet"
            fate.loc[doublet, "exclusion_reason"] = "predicted_doublet"
        fate.to_csv(self.reports / f"{dataset}_cell_fate.csv", index=False)

    def _parse(self, dataset: str):
        print(f"\n[{dataset}] download + parse source")
        builder = importlib.import_module(f"builders.{dataset}")
        adata = builder.build(self.raw, self.aux)
        # Source matching is complete at this point; enforce unique model IDs
        # before annotations/QC while retaining original source-ID columns.
        adata.obs_names_make_unique()
        adata.var_names_make_unique()
        ensure_counts(adata, dataset)
        print(
            f"[INFO] {dataset} parsed/source-selected cells: "
            f"{adata.n_obs:,} cells x {adata.n_vars:,} genes"
        )
        self._save(adata, dataset, "parsed")

    def _annotate(self, dataset: str):
        print(f"[{dataset}] attach model-facing annotations")
        adata = self._load(dataset, "qc")
        adata = annotate_dataset(adata, dataset)

        # Persist a compact, human-auditable map of every source label that
        # contributed to model-facing broad labels.  This makes odd
        # dataset-specific mappings visible before any cells are discarded.
        audit_columns = [
            c for c in (
                "Major_resolution_source",
                "Major_source_label",
                "Celltype_training",
                "Celltype_provenance",
            ) if c in adata.obs
        ]
        if audit_columns:
            audit = (
                adata.obs[audit_columns]
                .astype("object")
                .fillna("<NA>")
                .groupby(audit_columns, dropna=False)
                .size()
                .reset_index(name="n_cells")
                .sort_values("n_cells", ascending=False)
            )
            audit.to_csv(
                self.reports / f"{dataset}_annotation_resolution.csv",
                index=False,
            )

        missing_major = ~adata.obs["eligible_broad"]
        if missing_major.any():
            print(
                f"[INFO] {dataset}: retaining {int(missing_major.sum()):,} "
                "QC-passed cells in the audit artifact; they are excluded only "
                "from model-ready output because no supported source broad "
                "ground truth is available"
            )

        if dataset in CAF_DATASETS:
            unexpected = set(
                adata.obs["Celltype_training"].dropna().astype(str)
            ) - CAF_TARGET_LABELS
            if unexpected:
                raise ValueError(
                    f"{dataset}: unexpected CAF target labels: {sorted(unexpected)}"
                )

        self._save(adata, dataset, "annotated")
        self._write_cell_fate(dataset, adata)

    def _qc(self, dataset: str, seed: int):
        print(f"[{dataset}] quality control")
        adata = self._load(dataset, "parsed")
        before_cells = adata.n_obs

        if dataset in EXTRA_DATASETS:
            adata, qc_meta = scdiffusion_qc(adata)
        else:
            adata, qc_meta = percentile_qc(adata)

            if dataset in TEST_DATASETS:
                # Thesis policy: test doublets are annotated but not removed.
                adata = run_scrublet(
                    adata,
                    expected_doublet_rate=0.06,
                    seed=seed,
                    remove=False,
                )
                qc_meta["scrublet"] = "scored_and_retained"
                qc_meta["scrublet_expected_rate"] = 0.06

            if dataset in CAF_DATASETS:
                qc_meta["task"] = "CAF-specific tissue-held-out reconstruction"

        retained = (adata.n_obs / before_cells * 100.0) if before_cells else 0.0
        print(
            f"[INFO] {dataset} QC: {before_cells:,} -> {adata.n_obs:,} cells "
            f"({retained:.2f}% retained); genes={adata.n_vars:,}"
        )
        self._save(adata, dataset, "qc", qc_meta)

    def _normalize(self, dataset: str):
        print(f"[{dataset}] normalize + log1p")
        adata = self._load(dataset, "annotated")
        eligible = adata.obs["eligible_broad"].astype(bool)
        adata = adata[eligible].copy()
        if adata.n_obs == 0:
            raise ValueError(f"{dataset}: no model-eligible cells after annotation")
        adata = normalize_log(adata)
        adata = finalize_schema(
            adata,
            dataset,
            {
                "annotation_policy": (
                    "source-backed broad labels only; source-coarse ground truth "
                    "is retained in Celltype_training; manual labels are not used"
                )
            },
        )
        self._save(adata, dataset, "normalized")

    def _run_dataset(
        self,
        dataset: str,
        force: bool,
        from_stage: str | None,
        seed: int,
    ) -> None:
        if force:
            self.checkpoints.clear(dataset)
        if from_stage:
            self.checkpoints.invalidate_from(dataset, from_stage)

        if not self.checkpoints.done(dataset, "parsed"):
            self._parse(dataset)

        # Training QC is applied once after the five sources are integrated.
        if dataset in TRAIN_DATASETS:
            return

        if not self.checkpoints.done(dataset, "qc"):
            self._qc(dataset, seed)

        if not self.checkpoints.done(dataset, "annotated"):
            self._annotate(dataset)

        if not self.checkpoints.done(dataset, "normalized"):
            self._normalize(dataset)

    def _write_train(self, seed: int) -> None:
        checkpoint_name = "__TRAIN_INTEGRATION__"
        if self.checkpoints.done(checkpoint_name, "written"):
            subtype_dir = self.output / "Subtype"
            if not subtype_outputs_complete(subtype_dir):
                output = self.output / OUTPUT_PATHS["train"]
                print(
                    "[Subtype] derived files are missing; regenerate them "
                    "from the completed integrated training file",
                    flush=True,
                )
                source = ad.read_h5ad(output, backed="r")
                try:
                    write_subtype_splits(source, subtype_dir)
                finally:
                    source.file.close()
            return

        if not self.checkpoints.done(checkpoint_name, "qc"):
            print(
                "\n[Integrated_training] integrate five training datasets",
                flush=True,
            )
            components = []
            for dataset in TRAIN_DATASETS:
                if not self.checkpoints.done(dataset, "parsed"):
                    raise RuntimeError(f"{dataset}: parsed stage is missing")
                print(
                    f"[Integrated_training] load {dataset} count matrix",
                    flush=True,
                )
                # Parsed files contain X, counts and raw_counts as identical count
                # matrices. The integration/QC step needs only one of them.
                component = self._load_counts_only(dataset, "parsed")
                component.obs["source_cell_id"] = component.obs_names.astype(str)
                component.obs_names = pd.Index(
                    [f"{dataset}:{value}" for value in component.obs_names.astype(str)]
                )
                components.append(component)

            print(
                "[Integrated_training] concatenate shared genes",
                flush=True,
            )
            train = concat_intersection(components)
            components.clear()
            del component
            gc.collect()
            all_ids = train.obs_names.astype(str).copy()
            print(
                f"[INFO] Integrated training before QC: "
                f"{train.n_obs:,} cells x {train.n_vars:,} shared genes",
                flush=True,
            )

            train, qc_meta = percentile_qc(train)
            print(
                f"[INFO] Integrated training after QC: "
                f"{train.n_obs:,} cells x {train.n_vars:,} genes",
                flush=True,
            )
            print("[Integrated_training] run Scrublet", flush=True)
            train = run_scrublet(
                train,
                expected_doublet_rate=0.06,
                seed=seed,
                remove=False,
            )
            doublet_ids = train.obs_names[
                train.obs["predicted_doublets"]
            ].astype(str)
            train = train[~train.obs["predicted_doublets"]].copy()
            print(
                f"[INFO] Integrated training after Scrublet: "
                f"{train.n_obs:,} cells x {train.n_vars:,} genes",
                flush=True,
            )

            self._save(
                train,
                checkpoint_name,
                "qc",
                {
                    **qc_meta,
                    "scrublet": "removed",
                    "scrublet_expected_rate": 0.06,
                },
            )
        else:
            print(
                "\n[Integrated_training] resume from saved QC/Scrublet checkpoint",
                flush=True,
            )
            train = self._load(checkpoint_name, "qc")
            all_ids, doublet_ids = self._integrated_fate_inputs()

        if not self.checkpoints.done(checkpoint_name, "annotated"):
            train = self._attach_train_source_annotations(train)
        else:
            train = self._load(checkpoint_name, "annotated")

        observed_major = set(train.obs["Celltype_training"].dropna().astype(str))
        if MAIN_MAJOR_LABELS - observed_major:
            raise ValueError(
                f"Integrated_training is missing broad labels: "
                f"{sorted(MAIN_MAJOR_LABELS - observed_major)}"
            )

        if not self.checkpoints.done(checkpoint_name, "annotated"):
            # Validate before marking the annotation stage successful.
            self._save(train, checkpoint_name, "annotated")
            self._write_cell_fate(
                "Integrated_training",
                train,
                all_ids=all_ids,
                doublet_ids=doublet_ids,
            )

        if not self.checkpoints.done(checkpoint_name, "normalized"):
            pending = ~train.obs["eligible_broad"].astype(bool)
            if pending.any():
                print(
                    f"[INFO] Integrated_training: {int(pending.sum()):,} "
                    "QC-passed cells retained for audit but excluded from "
                    "model output",
                    flush=True,
                )
            train = train[~pending].copy()
            print("[Integrated_training] normalize + log1p", flush=True)
            train = normalize_log(train)

            # STOP preprocessing here. HVG, DGE, balancing, Harmony and all
            # model feature engineering are scLightGAT responsibilities.
            train = finalize_schema(
                train,
                "Integrated_training",
                {
                    "components": TRAIN_DATASETS,
                    "full_gene_space": True,
                    "feature_selection": "deferred_to_scLightGAT",
                    "scrublet_expected_rate": 0.06,
                },
            )
            self._save(train, checkpoint_name, "normalized")
        else:
            train = self._load(checkpoint_name, "normalized")

        # Reapply the schema at the write boundary so stale manual labels do
        # not leak into model-ready data.
        train = finalize_schema(
            train,
            "Integrated_training",
            {
                "components": TRAIN_DATASETS,
                "full_gene_space": True,
                "feature_selection": "deferred_to_scLightGAT",
                "scrublet_expected_rate": 0.06,
            },
        )
        output = self.output / OUTPUT_PATHS["train"]
        n_obs, n_vars = train.n_obs, train.n_vars
        atomic_write_h5ad(train, output)
        # The integrated object is large.  Release it before materializing each
        # subtype slice, then stream the just-written source in backed mode.
        del train
        gc.collect()
        source = ad.read_h5ad(output, backed="r")
        try:
            write_subtype_splits(source, self.output / "Subtype")
        finally:
            source.file.close()
        self.checkpoints.mark(
            checkpoint_name,
            "written",
            output,
            {"n_obs": n_obs, "n_vars": n_vars},
        )

    def _write_one(self, dataset: str) -> None:
        if self.checkpoints.done(dataset, "written"):
            return

        print(f"[{dataset}] write model-ready H5AD", flush=True)
        adata = self._load(dataset, "normalized")

        # Keep each independent test in its full post-QC normalized gene space.
        # scLightGAT itself must compute the train/test intersection separately
        # for the HVG and DGE branches.  Pre-projecting a test file to the 880
        # training DGE genes destroys the C-DVAE input space.


        adata = finalize_schema(adata, dataset)
        output = self.output / OUTPUT_PATHS[dataset]
        atomic_write_h5ad(adata, output)
        self.checkpoints.mark(
            dataset,
            "written",
            output,
            {"n_obs": adata.n_obs, "n_vars": adata.n_vars},
        )
        print(f"[{dataset}] write complete: {output}", flush=True)

    def _write_report(self, datasets: list[str]) -> None:
        print("[Report] update final_outputs.csv", flush=True)
        rows = []

        train_path = self.output / OUTPUT_PATHS["train"]
        if self.checkpoints.done("__TRAIN_INTEGRATION__", "written"):
            train = ad.read_h5ad(train_path, backed="r")
            rows.append(
                {
                    "dataset": "Integrated_training",
                    "n_cells": train.n_obs,
                    "n_genes": train.n_vars,
                    "celltype_nonnull": int(
                        train.obs["Celltype_training"].notna().sum()
                    ),
                    "output": str(train_path),
                }
            )
            train.file.close()

        for dataset in datasets:
            if dataset in TRAIN_DATASETS:
                continue
            output = self.output / OUTPUT_PATHS[dataset]
            if not self.checkpoints.done(dataset, "written"):
                continue
            adata = ad.read_h5ad(output, backed="r")
            rows.append(
                {
                    "dataset": dataset,
                    "n_cells": adata.n_obs,
                    "n_genes": adata.n_vars,
                    "celltype_nonnull": int(
                        adata.obs["Celltype_training"].notna().sum()
                    ),
                    "output": str(output),
                }
            )
            adata.file.close()

        pd.DataFrame(rows).to_csv(
            self.reports / "final_outputs.csv",
            index=False,
        )

    def build(
        self,
        datasets: list[str],
        keep_raw: bool = False,
        force: bool = False,
        from_stage: str | None = None,
        seed: int = 0,
    ) -> int:
        requested_train = any(dataset in TRAIN_DATASETS for dataset in datasets)
        wants_train = requested_train

        # Tests no longer depend on a pre-truncated training feature vocabulary.
        # Each output is reconstructed independently in full gene space.
        runset = list(dict.fromkeys(datasets))
        if wants_train:
            runset = list(dict.fromkeys([*runset, *TRAIN_DATASETS]))

        if wants_train and force:
            self.checkpoints.clear("__TRAIN_INTEGRATION__")
        elif wants_train and from_stage is not None:
            self.checkpoints.invalidate_from("__TRAIN_INTEGRATION__", from_stage)

        requested_set = set(datasets)
        for dataset in runset:
            try:
                # Dependency datasets are automatically added to the runset.
                # --force/--from-stage should not destroy their valid checkpoints
                # unless the user explicitly requested that dataset/group.
                invalidate_this_dataset = dataset in requested_set
                self._run_dataset(
                    dataset,
                    force=force and invalidate_this_dataset,
                    from_stage=from_stage if invalidate_this_dataset else None,
                    seed=seed,
                )
            except Exception as exc:
                print(f"\n[ERROR] {dataset}: {exc}")
                traceback.print_exc()
                return 2

        # Independent datasets are already normalized at this point.  Write
        # them before the much larger training integration so a failed/aborted
        # training job never hides completed test, extra or CAF outputs.
        for dataset in runset:
            if dataset not in TRAIN_DATASETS:
                self._write_one(dataset)

        if any(dataset not in TRAIN_DATASETS for dataset in runset):
            self._write_report(list(ALL_LOGICAL_DATASETS))

        if wants_train:
            self._write_train(seed)

        self._write_report(list(ALL_LOGICAL_DATASETS))

        if not keep_raw:
            if set(datasets) == set(ALL_LOGICAL_DATASETS):
                print("\nCleaning downloaded raw/source cache after successful full build...")
                shutil.rmtree(self.raw, ignore_errors=True)
                self.raw.mkdir(parents=True, exist_ok=True)
            else:
                print(
                    "\nPartial build completed; raw cache is retained so unrelated "
                    "interrupted downloads remain resumable. Use `python main.py clean` "
                    "when you want to remove downloaded sources."
                )

        print(f"\nDone. Output root: {self.output}")
        return 0

    def print_status(self) -> None:
        for dataset in ALL_LOGICAL_DATASETS + ["__TRAIN_INTEGRATION__"]:
            data = self.checkpoints.read(dataset)
            if data.get("code_fingerprint") != self.checkpoints.fingerprint:
                print(f"{dataset:22s} (stale; rebuild required)")
                continue
            completed = [
                stage
                for stage in STAGES
                if data.get("stages", {}).get(stage, {}).get("ok")
            ]
            text = " -> ".join(completed) if completed else "(not started)"
            print(f"{dataset:22s} {text}")

    def clean(self, intermediates: bool = False) -> None:
        shutil.rmtree(self.raw, ignore_errors=True)
        self.raw.mkdir(parents=True, exist_ok=True)

        if intermediates:
            shutil.rmtree(self.intermediate, ignore_errors=True)
            self.intermediate.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(self.aux, ignore_errors=True)
            self.aux.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(self.reports, ignore_errors=True)
            self.reports.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(self.checkpoints.root, ignore_errors=True)
            self.checkpoints.root.mkdir(parents=True, exist_ok=True)
