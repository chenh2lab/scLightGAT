"""Resource tracking and visualization helpers for annotation benchmarks."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from utils.dataset import prepare_umap


class MemoryTracker:
    """Sample peak RAM and NVIDIA VRAM for this process and its children."""

    def __init__(self, sample_interval=1.0):
        self.sample_interval = sample_interval
        self.root_pid = os.getpid()
        self._stop = False
        self._thread = None
        self._ram = []
        self._vram = []

    def _process_ids(self):
        pids = {self.root_pid}
        frontier = [self.root_pid]
        while frontier:
            pid = frontier.pop()
            path = f"/proc/{pid}/task/{pid}/children"
            try:
                children = [int(value) for value in Path(path).read_text().split()]
            except (OSError, ValueError):
                children = []
            for child in children:
                if child not in pids:
                    pids.add(child)
                    frontier.append(child)
        return pids

    @staticmethod
    def _ram_mb(pids):
        total_kb = 0
        for pid in pids:
            try:
                for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        total_kb += int(line.split()[1])
                        break
            except (OSError, ValueError):
                pass
        return total_kb / 1024.0

    @staticmethod
    def _vram_mb(pids):
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi", "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            return 0.0
        total = 0.0
        for line in output.splitlines():
            try:
                pid_text, memory_text = [part.strip() for part in line.split(",")]
                if int(pid_text) in pids:
                    total += float(memory_text)
            except (ValueError, TypeError):
                pass
        return total

    def _sample(self):
        while not self._stop:
            pids = self._process_ids()
            self._ram.append(self._ram_mb(pids))
            self._vram.append(self._vram_mb(pids))
            time.sleep(self.sample_interval)

    def start(self):
        self._stop = False
        self._ram = []
        self._vram = []
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        pids = self._process_ids()
        self._ram.append(self._ram_mb(pids))
        self._vram.append(self._vram_mb(pids))
        return {
            "peak_ram_mb": round(max(self._ram, default=0.0), 2),
            "peak_vram_mb": round(max(self._vram, default=0.0), 2),
        }


def _plot_umap(adata, color_key, output_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc.pl.umap(
        adata, color=color_key, show=False, frameon=False, title=title, ax=ax
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_and_save_umap(
    adata, pred_key, test_batch, test_celltype, seed, output_dir, tool_name
):
    """Compute the test-data UMAP and save coordinates plus applicable plots."""
    output_dir = Path(output_dir)
    vis = prepare_umap(adata, seed)
    adata.obsm["X_umap"] = np.asarray(vis.obsm["X_umap"])

    coordinates = pd.DataFrame(
        adata.obsm["X_umap"], index=adata.obs_names, columns=["UMAP1", "UMAP2"]
    )
    coordinates.index.name = "barcode"
    coordinates["prediction"] = adata.obs[pred_key].astype(str).to_numpy()
    if test_celltype is not None:
        coordinates["celltype"] = adata.obs[test_celltype].astype(str).to_numpy()
    if test_batch is not None:
        coordinates["batch"] = adata.obs[test_batch].astype(str).to_numpy()
    coordinates.to_csv(output_dir / "umap_coordinates.csv")

    _plot_umap(
        adata, pred_key,
        output_dir / f"{tool_name}_umap_prediction.png",
        f"{tool_name} prediction",
    )
    if test_celltype is not None:
        _plot_umap(
            adata, test_celltype,
            output_dir / f"{tool_name}_umap_celltype.png",
            "Ground truth",
        )
    if test_batch is not None:
        _plot_umap(
            adata, test_batch,
            output_dir / f"{tool_name}_umap_batch.png",
            "Batch",
        )

    if test_celltype is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sc.pl.umap(
            adata, color=test_celltype, show=False, frameon=False,
            title="Ground truth", ax=axes[0]
        )
        sc.pl.umap(
            adata, color=pred_key, show=False, frameon=False,
            title=f"{tool_name} prediction", ax=axes[1]
        )
        fig.savefig(
            output_dir / f"{tool_name}_umap_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def save_confusion_heatmap(confusion, output_path, title):
    """Save a row-normalized heatmap annotated with raw cell counts."""
    values = confusion.to_numpy(dtype=float)
    row_sums = values.sum(axis=1, keepdims=True)
    normalized = np.divide(
        values, row_sums, out=np.zeros_like(values), where=row_sums != 0
    )
    size = max(7, min(18, 0.65 * len(confusion) + 4))
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        normalized,
        annot=values.astype(int),
        fmt="d",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=confusion.columns,
        yticklabels=confusion.index,
        cbar_kws={"label": "Row proportion"},
        ax=ax,
    )
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
