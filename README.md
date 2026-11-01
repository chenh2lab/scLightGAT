# scLightGAT: A LightGBM-based Framework Integrating C-DVAE and GATs for Robust Single-cell RNA-seq Cell Type Annotation

scLightGAT is a supervised reference-to-query framework for single-cell RNA-seq cell-type annotation. It combines a contrastive denoising variational autoencoder (C-DVAE), a LightGBM classifier, and a graph attention network (GAT), allowing expression-derived features, discriminative marker genes, and local cell-neighborhood information to contribute to the final prediction.

During reference training, scLightGAT selects differentially expressed marker genes and up to 3,000 highly variable genes from the labeled reference data only. The C-DVAE learns a 128-dimensional latent representation using reconstruction, KL divergence, and supervised contrastive objectives. LightGBM combines marker-gene expression with this latent representation to produce initial class probabilities; out-of-fold probabilities are used when fitting the downstream GAT. A two-layer residual GAT then refines the initial predictions using marker-gene expression, latent features, and LightGBM probabilities on a 10-nearest-neighbor graph built in PCA space, with Harmony correction when valid batch metadata are supplied.

The fitted feature schemas and model components are reused for independent query datasets. Query genes are aligned to the reference DEG and HVG feature sets, missing genes are imputed with reference-derived means, and each query graph is constructed independently from query expression and optional batch metadata without using query cell-type labels. The initial LightGBM labels are retained in `adata.obs["scLightGAT_rawpred"]`, while the GAT-refined annotations and class probabilities are stored in `adata.obs["scLightGAT_pred"]` and `adata.obsm["annotation_scores"]`.

![Overview of the scLightGAT architecture](docs/overview.jpg)

## Prerequisites

- Linux with Bash
- Git
- A Conda distribution, such as Miniconda or Anaconda
- Python 3.10, installed through the Conda environment below
- Optional: a CUDA-capable NVIDIA GPU with a compatible NVIDIA driver for GPU acceleration. CPU execution is supported but may be slower.

### Input requirements

- A labeled reference dataset (`--train_path`) and one or more query datasets (`--test_path`) in AnnData (`.h5ad`) format. `--test_path` may point to a single file or a directory of `.h5ad` files.
- Quality-controlled, library-size-normalized, and `log1p`-transformed expression in `adata.X` and `adata.layers["log_transformed"]`. The reference dataset must also contain raw counts in `adata.layers["counts"]`.
- A reference cell-type column in `adata.obs`, specified with `--train_celltype`, containing at least two classes with at least two cells per class. Query labels are optional and are used only for evaluation.
- Batch columns are optional and can be specified with `--train_batch` and `--test_batch`; use `None` when batch information is unavailable.

## Installation

scLightGAT is implemented in Python. Clone the repository and create the Conda environment as follows:

```bash
git clone https://github.com/chenh2lab/scLightGAT
cd scLightGAT
conda create --name scLightGAT python=3.10
conda activate scLightGAT
python -m pip install -r requirements.txt
python -m pip check
```

## Data Availability

The datasets used to train and evaluate scLightGAT are summarized below. The five training datasets were integrated into a reference comprising 236,975 cells, while the testing datasets were retained as independent query datasets. Cell counts refer to the cells retained in the final model-ready datasets.

<table>
<thead>
  <tr>
    <th>Dataset Role</th>
    <th>Biological Context</th>
    <th>Source / Accession</th>
    <th>Cells in Dataset</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="5">Training Datasets</td>
    <td>NSCLC</td>
    <td>GSE131907</td>
    <td>73,593</td>
  </tr>

  <tr>
      <td>Breast cancer</td>
      <td>GSE176078</td>
      <td>83,472</td>
  </tr>

  <tr>
      <td>Hepatocellular carcinoma</td>
      <td>GSE140228</td>
      <td>17,182</td>
  </tr>

  <tr>
      <td>Colorectal cancer</td>
      <td>GSE132465</td>
      <td>33,092</td>
  </tr>

  <tr>
      <td>Prostate Cancer</td>
      <td>GSE141445</td>
      <td>29,636</td>
  </tr>

  <tr>
    <td rowspan="7">Testing Datasets</td>
    <td>NSCLC</td>
    <td>GSE153935</td>
    <td>4,844</td>
  </tr>

  <tr>
    <td>Melanoma</td>
    <td>GSE115978</td>
    <td>4,634</td>
  </tr>

  <tr>
    <td>Melanoma immune cells</td>
    <td>GSE123139</td>
    <td>7,903</td>
  </tr>

  <tr>
    <td>Colorectal cancer</td>
    <td>GSE166555</td>
    <td>35,382</td>
  </tr>

  <tr>
    <td>Sorted PBMC</td>
    <td>Zheng</td>
    <td>17,385</td>
  </tr>

  <tr>
    <td>Pulmonary fibrosis lung</td>
    <td>GSE135893</td>
    <td>113,002</td>
  </tr>

  <tr>
    <td>Tabula Sapiens</td>
    <td>GSE201333</td>
    <td>49,357</td>
  </tr>

  <tr>
    <td>CAF Datasets</td>
    <td>Pan-cancer CAF atlas</td>
    <td>GSE210347</td>
    <td>73,402<br>(51,097 reference / 22,305 query)</td>
  </tr>
  </tbody>
</table>

The subtype benchmarks were derived from the integrated reference using predefined donor-level splits. The CAF benchmark was derived from GSE210347 using a predefined tissue-level split. Within each task, the reference and query subsets are mutually exclusive.

### Option 1: Download the processed datasets

The processed scLightGAT datasets are available on [Hugging Face](https://huggingface.co/datasets/Alfiechuang/scLightGAT).

After installing `huggingface_hub` (`python -m pip install huggingface_hub`), the complete processed dataset repository can be downloaded with Python:

```python
from huggingface_hub import snapshot_download

data_dir = snapshot_download(
    repo_id="Alfiechuang/scLightGAT",
    repo_type="dataset",
    local_dir="scLightGAT_data",
)
```

### Option 2: Rebuild the datasets locally

The preprocessing pipeline downloads the publicly available source data, performs quality control and cell-type label harmonization, and generates the `.h5ad` files used in the experiments. See [`preprocessing/README.md`](preprocessing/README.md) for setup and execution instructions.

## Usage

Run scLightGAT from the repository root after completing the installation steps above.

If Conda is not available in `PATH`, specify its installation directory:

```bash
export CONDA_BASE="<conda_path>"
```

Run scLightGAT with the following command. Replace each value enclosed in angle
brackets (`<...>`) with the corresponding value for your dataset:

```bash
bash run_script.sh \
  --train_path "<reference_h5ad>" \
  --test_path "<query_h5ad_or_directory>" \
  --save_path "<output_directory>" \
  --train_celltype "<reference_celltype_key>" \
  --test_celltype "<query_celltype_key_or_None>" \
  --train_batch "<reference_batch_key_or_None>" \
  --n_runs "<number_of_runs>"
```

Use `--test_batch` to specify the default batch column for query datasets:

```bash
--test_batch "<query_batch_key>" \
```

When `--test_path` is a directory, use `--test_batch_map` to override the default for individual query files:

```bash
--test_batch_map "dataset_a=batch_key,dataset_b=sample_key,dataset_c=None" \
```

Each mapping key must match an `.h5ad` file name without its extension. Files omitted from the map use the value of `--test_batch`; omit both options when query batch information is unavailable.

`--train_path`, `--test_path`, and `--save_path` are required. `--test_path` may point to one `.h5ad` file or a directory of `.h5ad` files; in the latter case, the reference is trained once per run and applied to every query file. The optional defaults are `--train_celltype Celltype_training`, `--test_celltype Celltype_training`, `--train_batch Data_source`, no query batch column, and `--n_runs 1`. Set `--test_celltype None` for an unlabeled query; predictions are still generated, but evaluation metrics are skipped.

## Reproducing the Experiments

### Tested configuration

- OS: NVIDIA DGX OS
- CPU: Cortex-X925 / Cortex-A725
- GPU: NVIDIA GB10
- NVIDIA driver: 580.126.09
- Python: 3.10.20
- PyTorch: 2.11.0+cu130

### Shared dataset parameters

The experiments use the following shared metadata keys. Use these values wherever the commands below contain the corresponding placeholders:

| `<task_name>` | `<train_celltype_key>` | `<test_celltype_key>` | `<train_batch_key>` | `<test_batch_key>` |
| --- | --- | --- | --- | --- |
| `broad` | `Celltype_training` | `Celltype_training` | `Data_source` | `None` |
| `CAF` | `Celltype_training` | `Celltype_training` | `SampleID` | `None` |
| `subtype` | `Celltype_subtraining` | `Celltype_subtraining` | `Data_source` | `Data_source` |

For the broad benchmark, use `--test_batch_map "GSE115978=samples,GSE123139=Sample.Short.Name,GSE153935=orig.ident,GSE166555=sample_id,Zhengsorted=None,lung_full=None,sapiens_full=donor"`. Use an empty value for `<test_batch_map>` in the CAF and subtype benchmarks.

### Experiment list

| Experiment | Description |
| --- | --- |
| [Main experiment](#main-experiment) | Compares the annotation performance and runtime of scLightGAT with existing cell-type annotation methods across broad, CAF, and subtype benchmarks. |
| [Ablation study](#ablation-study) | Evaluates the contributions of C-DVAE and GAT by comparing the full model with No C-DVAE, No GAT, and LightGBM-only variants. |

#### Main experiment

The main experiment uses the integrated reference and the independent query datasets described in the Data Availability section. We compare scLightGAT with seven reference-based cell-type annotation methods: CellTypist, CHETAH, Seurat, SingleR, scANVI, scGraphformer, and scGPT. The comparison covers seven broad cell-type datasets, one CAF task, and five subtype tasks.

Before running the experiment, create the method-specific Conda environments by following the instructions in [`benchmark/README.md`](benchmark/README.md). Run scLightGAT first and then the comparison methods for each task, saving their outputs under the same task-specific result directory so that they reuse the same five-run seed schedule.

Run scLightGAT:

```bash
bash run_script.sh \
  --train_path "<train_path>" \
  --test_path "<test_path>" \
  --save_path "<result_path>" \
  --train_celltype "<train_celltype_key>" \
  --test_celltype "<test_celltype_key>" \
  --train_batch "<train_batch_key>" \
  --test_batch "<test_batch_key>" \
  --test_batch_map "<test_batch_map>" \
  --n_runs 5
```

Run the comparison methods with the same values:

```bash
bash benchmark/run_script.sh \
  --train_path "<train_path>" \
  --test_path "<test_path>" \
  --save_path "<result_path>" \
  --train_celltype "<train_celltype_key>" \
  --test_celltype "<test_celltype_key>" \
  --train_batch "<train_batch_key>" \
  --test_batch "<test_batch_key>" \
  --test_batch_map "<test_batch_map>" \
  --n_runs 5
```

Repeat both commands with the following paths:

- **Broad:** `<data_root>/Integrated_training/train.h5ad` to `<data_root>/Independent_testing`, saved to `<result_root>`.
- **CAF:** `<data_root>/caf.data/caf_train.h5ad` to `<data_root>/caf.data/caf_test.h5ad`, saved to `<result_root>/CAF`.
- **Subtype:** `<data_root>/Subtype/<broad_type>_train.h5ad` to `<data_root>/Subtype/<broad_type>_test.h5ad`, saved to `<result_root>/Subtype/<broad_type>`, for `B_cell`, `CD4_T_cell`, `CD8_T_cell`, `DC`, and `Plasma_cell`.

All methods use the same reference and query datasets. Query cell-type labels are used exclusively for evaluation and are excluded from feature selection, model fitting, and prediction.

After scLightGAT and all comparison methods have finished, activate the `scLightGAT` environment and generate the metric, runtime, and UMAP figures. For each method, the UMAP script selects the completed run with the highest balanced accuracy.

```bash
conda activate scLightGAT

python3 bar_chart.py \
  --results_dir "<result_root>" \
  --sclightgat_results_dir "<result_root>" \
  --out_dir "<plot_root>/bar_chart"

python3 plot_umap.py \
  --test_path \
    "<data_root>/Independent_testing" \
    "<data_root>/caf.data/caf_test.h5ad" \
  --results_dir "<result_root>" \
  --sclightgat_results_dir "<result_root>" \
  --subtype_results_dir "<result_root>/Subtype" \
  --subtype_sclightgat_results_dir "<result_root>/Subtype" \
  --subtype_test_dir "<data_root>/Subtype" \
  --out_dir "<plot_root>/umap" \
  --subtype_out_dir "<plot_root>/umap/Subtype"
```

#### Ablation study

This experiment evaluates three scLightGAT variants while keeping the remaining preprocessing, training, and evaluation settings unchanged:

- **No C-DVAE (`No_CDVAE`):** Removes the C-DVAE representation from the full pipeline.
- **No GAT (`No_GAT`):** Removes graph-attention refinement from the full pipeline.
- **LightGBM only (`Only_LightGBM`):** Uses the marker-gene-based LightGBM classifier without C-DVAE or GAT.

Run the main experiment first. For each broad, CAF, and subtype task, copy the corresponding five-run seed schedule and then run all three ablation variants from the repository root:

```bash
mkdir -p "<ablation_result_path>/.benchmark_state/seeds"
cp "<full_result_path>/.benchmark_state/seeds"/run_{1..5}.json \
  "<ablation_result_path>/.benchmark_state/seeds/"

bash experiments/ablation/run_script.sh \
  --train_path "<train_path>" \
  --test_path "<test_path>" \
  --save_path "<ablation_result_path>" \
  --train_celltype "<train_celltype_key>" \
  --test_celltype "<test_celltype_key>" \
  --train_batch "<train_batch_key>" \
  --test_batch "<test_batch_key>" \
  --test_batch_map "<test_batch_map>" \
  --n_runs 5
```

Use the same dataset paths and metadata keys as the corresponding main-experiment task. Ablation outputs must remain under `experiments/ablation/result`: use that directory for the broad task, its `CAF/` subdirectory for CAF, and `Subtype/<broad_type>/` for each subtype. Per-run results are stored below the `No_CDVAE`, `No_GAT`, and `Only_LightGBM` method directories.

After all ablation runs finish, generate the comparison figures separately:

```bash
conda activate scLightGAT

python3 experiments/ablation/bar_chart.py \
  --results_dir "<ablation_result_root>" \
  --sclightgat_results_dir "<result_root>" \
  --out_dir "<plot_root>/ablation/bar_chart"

python3 experiments/ablation/plot_umap.py \
  --results_dir "<ablation_result_root>" \
  --baseline_results_dir "<result_root>" \
  --out_dir "<plot_root>/ablation/umap"

python3 experiments/ablation/heatmap.py \
  --results_dir "<ablation_result_root>" \
  --sclightgat_results_dir "<result_root>" \
  --out_dir "<plot_root>/ablation/heatmap"
```

## Citation

Citation information for scLightGAT will be added once the associated manuscript is published or a DOI becomes available.