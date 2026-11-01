# Single-Cell Cell-Type Annotation Benchmark of scLightGAT

This directory provides the comparison-method benchmark used to evaluate scLightGAT against seven supervised, reference-based methods for single-cell RNA-seq cell-type annotation: CellTypist, CHETAH, Seurat, SingleR, scANVI, scGraphformer, and scGPT.

Each method uses a common command-line interface and follows the same reference-to-query protocol: one reference state is prepared per run and seed from a labeled reference and reused across query datasets, while query-specific mapping, adaptation, or inference generates a cell-type prediction for each query cell, together with class scores when supported. The benchmark supports repeated runs and evaluates overall and per-class classification performance, training and prediction runtime, and peak RAM/VRAM usage.

The framework standardizes data loading, input validation, seed management, execution, output organization, and downstream evaluation while allowing each method to retain its own documented preprocessing, feature-selection, training, query-adaptation, and inference workflow based on the corresponding publication, official implementation, tool defaults, or an explicitly documented benchmark choice. Shared data loading and validation are performed before timing, whereas method-specific preprocessing, training, query adaptation, and prediction are included in the corresponding measured calls. Downstream evaluation and visualization are excluded from runtime and memory measurements.

## Included Methods

Methods are grouped by their primary methodological approach. Because several methods combine multiple model architectures or training objectives, these categories describe their dominant design and are not necessarily mutually exclusive.

`Input Preprocessing` summarizes the method-specific workflow implemented by each wrapper. For CellTypist, SingleR, CHETAH, and scGraphformer, normalization and log1p transformation are applied only when `adata.X` is detected as raw counts; already processed expression is reused unchanged.

<table>
<thead>
  <tr>
    <th>Major Class</th>
    <th>Minor Class</th>
    <th>Name</th>
    <th>Input Preprocessing</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="4">Classical Annotation</td>
    <td>Classical Machine Learning</td>
    <td>CellTypist</td>
    <td>
      Normalize (10,000)<br>
      log1p<br>
      Remove non-expressed genes<br>
      Scale (Max 10)<br>
    </td>
  </tr>
  <tr>
    <td rowspan="3">Reference Mapping</td>
    <td>SingleR</td>
    <td>
      Normalize (10,000)<br>
      log1p<br>
      Shared-gene matching<br>
      Marker selection (Wilcoxon)<br>
    </td>
  </tr>
  <tr>
    <td>CHETAH</td>
    <td>
      Normalize (10,000)<br>
      log1p<br>
      Shared-gene matching<br>
      Discriminating genes (&le;200)<br>
    </td>
  </tr>
  <tr>
    <td>Seurat</td>
    <td>
      LogNormalize (10,000)<br>
      HVG (2000)<br>
      Scale (Max 10)<br>
    </td>
  </tr>
  <tr>
    <td rowspan="2">Deep Learning</td>
    <td>Generative Model</td>
    <td>scANVI</td>
    <td>
      HVG (2000)<br>
      Raw counts<br>
      Query gene alignment<br>
    </td>
  </tr>
  <tr>
    <td>GNN-based</td>
    <td>scGraphformer</td>
    <td>
      Normalize (10,000)<br>
      log1p<br>
      HVG (3500)<br>
      Query gene alignment<br>
      KNN graph (k=8)<br>
    </td>
  </tr>
  <tr>
    <td>Foundation Model</td>
    <td>Transcriptomic Foundation Model</td>
    <td>scGPT</td>
    <td>
      Filter genes (model vocabulary)<br>
      Normalize (10,000)<br>
      log1p<br>
      Binning (51)<br>
      Tokenization (Max 3001)<br>
    </td>
  </tr>

</tbody>
</table>

## Evaluation Protocol

Each run prepares a method-specific reference state from one labeled reference dataset and predicts cell types for either one query dataset or every `.h5ad` file directly contained in a query directory. When multiple query datasets are supplied, one reference state is prepared per run and seed, while query-specific mapping, adaptation, and prediction are performed independently. Query ground-truth labels are withheld during prediction, and each method must return one prediction for every query cell.

When query labels are available, the shared workflow reports accuracy, balanced accuracy, macro/micro/weighted precision, recall, and F1 score, one-vs-rest AUC, Matthews correlation coefficient, Cohen's kappa, per-class metrics, and confusion matrices. Dataset-specific label mappings are applied only where explicitly defined, and all primary classification metrics, AUCs, and confusion matrices use the resulting source-compatible predictions and class scores; exact-label agreement is retained as a separate audit metric. When method-provided class scores are unavailable, AUC is calculated from hard-prediction scores. When `--test_celltype None` is used, predictions and UMAP visualizations are generated without classification metrics.

Training and prediction are measured separately. Method-specific preprocessing and query adaptation are included in the corresponding measurement. Data-loading time is excluded, although the loaded input remains part of measured process RAM; downstream evaluation, result writing, and visualization are excluded from runtime and memory measurements. The benchmark launcher applies the same limit of 12 CPU workers or threads to every method. UMAP coordinates are computed from query expression for visualization only; they are not method-produced embeddings and are not used to calculate annotation metrics.

## Prerequisites

- Linux with Bash
- Git
- A Conda distribution, such as Miniconda or Anaconda; each environment installs its required Python version.

### Input requirements

The benchmark requires one labeled reference dataset and one or more query datasets in AnnData (`.h5ad`) format:

- `--train_path` must point to one reference `.h5ad` file.
- `--test_path` may point to one query `.h5ad` file or a directory containing query `.h5ad` files.
- Reference and query datasets must contain normalized, `log1p`-transformed expression in `adata.X` and gene identifiers in `adata.var_names`.
- Raw counts must be stored in `adata.layers["counts"]` when running the complete benchmark; this layer is required by scANVI and used by Seurat.
- `--train_celltype` must specify a reference-label column containing at least two non-empty cell types.
- `--test_celltype` specifies an optional query ground-truth column used only for evaluation.
- `--train_batch` and `--test_batch` specify optional reference and query batch columns.
- `--test_batch_map` may specify different batch columns for individual files in a query directory.
- Use `None` when `--test_celltype`, `--train_batch`, or `--test_batch` is unavailable.

Each query dataset must contain at least three cells and at least two expressed genes for the shared UMAP workflow.

## Environment Installation

This benchmark uses multiple Conda environments to isolate incompatible method-specific dependencies. Create all environments before running the benchmark launcher, which executes every included method in sequence. When running an individual wrapper directly, only that method's environment is required. The launcher activates the appropriate environment for each method; it does not create or install environments automatically.

The shared `sctools` environment supports CellTypist and scANVI. CHETAH, Seurat, SingleR, scGPT, and scGraphformer use separate environments because their verified R/Bioconductor, Python, PyTorch, or CUDA dependencies are not mutually compatible. CHETAH, Seurat, and SingleR execute their R implementations through Python wrappers so that all methods retain the same command-line interface, output structure, and evaluation pipeline.

Use the commands below to create the environments with the exact names expected by the benchmark launcher. The PyTorch commands target Linux with CUDA 13 and use official wheels available for both `aarch64` and `x86_64`.

Commands that reference project files assume the current directory is `benchmark/`.

Before installing or running the benchmark, set the following variable in every shell session to prevent user-site packages from leaking into the Conda environments:

```bash
export PYTHONNOUSERSITE=1
```

### sctools: CellTypist and scANVI

Install the official PyTorch 2.12.0 CUDA 13 build before the remaining packages:

```bash
conda create -n sctools -c conda-forge --strict-channel-priority \
  python=3.10.20 pip=26.2.1 -y
conda activate sctools

python -m pip install "torch==2.12.0" \
  --index-url https://download.pytorch.org/whl/cu130

python -c 'import platform, torch; device="cuda" if torch.cuda.is_available() else "cpu"; x=torch.ones(1, device=device); print(platform.machine(), torch.__version__, torch.version.cuda, device, x.item())'

python -m pip install \
  "numpy==1.26.4" "pandas==2.3.3" "scipy==1.11.4" \
  "scikit-learn==1.7.2" "h5py==3.16.0" \
  "anndata==0.11.4" "scanpy==1.9.8" \
  "matplotlib==3.10.9" "seaborn==0.13.2" \
  "umap-learn==0.5.12" \
  "celltypist==1.7.1" "scvi-tools==1.3.3"

python -m pip install \
  "jax==0.6.1" "jaxlib==0.6.1" "lightning==2.6.4" \
  "pytorch-lightning==2.6.1" "torchmetrics==1.2.1" \
  "mudata==0.3.8" "numba==0.65.1" "llvmlite==0.47.0" \
  "statsmodels==0.14.6"

python -m pip install --force-reinstall "charset-normalizer==3.4.4"

python -m pip check
```

For another CUDA version or a CPU-only system, replace only the PyTorch command
with a compatible build from the [official installation instructions](https://pytorch.org/get-started/previous-versions/) before installing the remaining pinned packages.

### CHETAH

```bash
conda create -n CHETAH \
  --copy --override-channels --strict-channel-priority \
  -c conda-forge -c bioconda \
  python=3.10.22 pip=26.2.1 \
  r-base=4.3.3 r-argparse=2.2.5 r-matrix=1.6_5 \
  r-data.table r-htmlwidgets r-plotly r-shiny \
  bioconductor-data-packages=20260207=hdfd78af_0 \
  bioconductor-genomeinfodbdata=1.2.11=r43hdfd78af_0 \
  bioconductor-chetah=1.18.0 \
  bioconductor-s4vectors=0.40.2 \
  bioconductor-singlecellexperiment=1.24.0 \
  bioconductor-summarizedexperiment=1.32.0 \
  bioconductor-genomicranges=1.54.1 \
  bioconductor-delayedarray=0.28.0 -y
conda activate CHETAH

python -m pip install \
  "numpy==1.26.4" "pandas==2.3.3" "scipy==1.15.3" \
  "scikit-learn==1.7.2" "h5py==3.16.0" \
  "anndata==0.11.4" "scanpy==1.11.5" \
  "matplotlib==3.10.9" "seaborn==0.13.2" "umap-learn==0.5.12"

python -m pip check
```

### Seurat

```bash
conda create -n Seurat -c conda-forge --strict-channel-priority \
  python=3.10.22 pip=26.2.1 \
  r-base=4.3.3 r-argparse=2.2.5 r-matrix=1.6_5 \
  r-seurat=5.3.0 r-seuratobject=5.2.0 -y
conda activate Seurat

python -m pip install \
  "numpy==1.26.4" "pandas==2.3.3" "scipy==1.15.3" \
  "scikit-learn==1.7.2" "h5py==3.16.0" \
  "anndata==0.11.4" "scanpy==1.11.5" \
  "matplotlib==3.10.9" "seaborn==0.13.2" "umap-learn==0.5.12"

python -m pip check
```

### SingleR

```bash
conda create -n SingleR \
  -c conda-forge -c bioconda --strict-channel-priority \
  python=3.10.22 pip=26.2.1 \
  r-base=4.4.3 r-argparse=2.3.1 r-matrix=1.7_6 r-biocmanager \
  bioconductor-singler=2.8.0 \
  bioconductor-singlecellexperiment=1.28.0 \
  bioconductor-summarizedexperiment=1.36.0 \
  bioconductor-s4vectors=0.44.0 \
  bioconductor-biocparallel=1.40.0 \
  bioconductor-scran=1.34.0 -y
conda activate SingleR

python -m pip install \
  "numpy==1.26.4" "pandas==2.3.3" "scipy==1.15.3" \
  "scikit-learn==1.7.2" "h5py==3.16.0" \
  "anndata==0.11.4" "scanpy==1.11.5" \
  "matplotlib==3.10.9" "seaborn==0.13.2" "umap-learn==0.5.12"

python -m pip check
```

### scGPT

Create `scGPT` by cloning `sctools` so that both environments retain the same
platform-compatible PyTorch build. The benchmark does not require `torchtext`
or an external `flash_attn` package.

```bash
conda create -n scGPT --clone sctools -y
conda activate scGPT

python -m pip install \
  "scipy==1.15.3" "anndata==0.11.4" "scanpy==1.11.5" \
  "scvi-tools==0.20.3" "cell-gears==0.0.2" \
  "datasets==2.21.0" "leidenalg==0.12.0" "numba==0.67.0" \
  "orbax==0.1.7" "scib==1.1.7" "scikit-misc==0.5.2" \
  "gdown==6.1.0"
python -m pip install --no-deps "scgpt==0.2.4"
python -m pip install --force-reinstall "charset-normalizer==3.4.4"

python -c 'import torch; print(torch.__version__, torch.version.cuda)'
PYTHONPATH=. python -c 'import sys; sys.path.insert(0, "scGPT"); import model_impl; print("scGPT compatibility imports OK")'
```

The benchmark launcher validates and downloads the official whole-human checkpoint
when it is missing. It can also be downloaded before running the launcher:

```bash
conda activate scGPT
python scGPT/download_checkpoint.py \
  --output_dir scGPT/models/scGPT_human
```

### scGraphformer

```bash
conda create -n scGraphformer -c conda-forge \
  python=3.10 git pip numpy=2.0.2 pandas=2.2.3 scipy=1.14.1 \
  scikit-learn=1.6.0 h5py=3.12.1 matplotlib=3.9.3 \
  seaborn=0.13.2 -y
conda activate scGraphformer

python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.11.0+cu130'
python -m pip install \
  anndata==0.11.1 scanpy==1.10.4 torch-geometric==2.8.0.post1

mkdir -p "$CONDA_PREFIX/src"
git clone https://github.com/xyfan22/scGraphformer.git \
  "$CONDA_PREFIX/src/scGraphformer"
git -C "$CONDA_PREFIX/src/scGraphformer" checkout --detach \
  75c76fc6b66195297a764d6e4569b3c01e084538

python -c 'from scGraphformer.scGraphformer import _official_model_class; cls = _official_model_class(); assert cls.__name__ == "scGraphformer"; print("scGraphformer model import OK")'
```

The benchmark's native `torch_sparse` fallback is restricted to
`aarch64`/`arm64`. On x86_64, install a `torch_sparse` build matching the
selected PyTorch and CUDA versions.

## Usage

### Running the Benchmark

Run the comparison-method benchmark from the repository root after completing
the installation steps above. The launcher runs the seven methods listed in
this README; run scLightGAT separately using the repository-root launcher
described in the main [`README.md`](../README.md).

The benchmark launcher initializes Conda using the path configured near the top
of [`benchmark/run_script.sh`](run_script.sh). Update that path when using a
different Conda installation.

Replace each value enclosed in angle brackets (`<...>`) with the corresponding
value for your dataset:

```bash
bash benchmark/run_script.sh \
  --train_path "<reference_h5ad>" \
  --test_path "<query_h5ad_or_directory>" \
  --save_path "<output_directory>" \
  --train_celltype "<reference_celltype_key>" \
  --test_celltype "<query_celltype_key_or_None>" \
  --train_batch "<reference_batch_key_or_None>" \
  --n_runs "<number_of_runs>"
```

If the current directory is `benchmark/`, use `bash run_script.sh` instead.

The launcher accepts the following query-batch parameters:

- `--test_batch "<query_batch_key_or_None>"` sets the default query batch column.
- `--test_batch_map "dataset_a=batch_key,dataset_b=sample_key,dataset_c=None"` overrides it for individual files when `--test_path` is a directory.

Each mapping key must match an `.h5ad` file name without its extension. Files omitted from the map use the value of `--test_batch`. When query batch information is unavailable, set `--test_batch None` and omit `--test_batch_map`.

`--train_path`, `--test_path`, and `--save_path` are required. `--test_path` may point to one `.h5ad` file or a directory of `.h5ad` files; in the latter case, one reference state is prepared per run and reused for every query file. The launcher defaults are `--train_celltype celltype`, `--test_celltype celltype`, `--train_batch batch`, `--test_batch batch`, and `--n_runs 5`. Set `--test_celltype None` for an unlabeled query; predictions are still generated, but evaluation metrics are skipped.
