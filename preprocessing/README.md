# scLightGAT Data Preprocessing

The datasets used by scLightGAT are assembled from multiple public sources, including the Gene Expression Omnibus (GEO) and data archives released with the corresponding publications. This pipeline automatically downloads the required files, performs standardized preprocessing, and harmonizes cell-type labels across sources. Training annotations are mapped to the predefined scLightGAT label set. Test annotations are mapped to the most specific supported class within the broad cell-type ontology; when a supported source annotation is coarser than this ontology, it is retained rather than inferred or subdivided. Coarse labels such as `T.cells` are not retained in the final training ground truth.

<table>
<thead>
  <tr>
    <th>Dataset Role</th>
    <th>Biological Context</th>
    <th>Source / Accession</th>
    <th>Cells (Filtered)</th>
    <th>Subtype</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="5">Training Datasets</td>
    <td>NSCLC</td>
    <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131907">GSE131907</a></td>
    <td>73,593</td>
    <td>Follicular B cells<br>Germinal B cells<br>MALT B cells<br>CD4+Tfh/Th cells<br>CD4+exhausted T cells<br>CD4+naive T cells<br>CD4+reg T cells<br>CD8+Naive T cells<br>CD8+exhausted T cells<br>DC<br>cDC<br>Plasma cells</td>
  </tr>

  <tr>
      <td>Breast cancer</td>
      <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176078">GSE176078</a></td>
      <td>83,472</td>
      <td>Memory B cells<br>Naive B cells<br>CD4+Tfh/Th cells<br>CD4+memory T cells<br>CD4+naive T cells<br>CD4+reg T cells<br>CD8+exhausted T cells<br>CD8+memory T cells<br>DC<br>cDC<br>Plasmablasts</td>
  </tr>

  <tr>
      <td>Hepatocellular carcinoma</td>
      <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140228">GSE140228</a></td>
      <td>17,182</td>
      <td>CD4+Tfh/Th cells<br>CD4+memory T cells<br>CD4+naive T cells<br>CD4+reg T cells<br>CD8+MAIT T cells<br>CD8+Naive T cells<br>CD8+exhausted T cells<br>CD8+memory T cells<br>DC<br>cDC<br>Plasma cells</td>
  </tr>

  <tr>
      <td>Colorectal cancer</td>
      <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132465">GSE132465</a></td>
      <td>33,092</td>
      <td>CD4+Tfh/Th cells<br>cDC<br>IgA+ Plasma<br>IgG+ Plasma</td>
  </tr>

  <tr>
      <td>Prostate cancer</td>
      <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE141445">GSE141445</a></td>
      <td>29,636</td>
      <td></td>
  </tr>

  <tr>
    <td rowspan="7">Test Datasets</td>
    <td>NSCLC</td>
    <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE153935">GSE153935</a></td>
    <td>4,844</td>
    <td></td>
  </tr>

  <tr>
    <td>Melanoma</td>
    <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115978">GSE115978</a></td>
    <td>4,634</td>
    <td></td>
  </tr>

  <tr>
    <td>Melanoma immune cells</td>
    <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE123139">GSE123139</a></td>
    <td>7,903</td>
    <td></td>
  </tr>

  <tr>
    <td>Colorectal cancer</td>
    <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE166555">GSE166555</a></td>
    <td>35,382</td>
    <td></td>
  </tr>

  <tr>
    <td>Sorted PBMC</td>
    <td><a href="https://doi.org/10.5281/zenodo.3357167">Zheng</a></td>
    <td>17,385</td>
    <td></td>
  </tr>

  <tr>
    <td>Pulmonary fibrosis lung</td>
    <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135893">GSE135893</a></td>
    <td>113,002</td>
    <td></td>
  </tr>

  <tr>
    <td>Tabula Sapiens</td>
    <td><a href="https://doi.org/10.6084/m9.figshare.25335436.v2">GSE201333</a><br>Download from scDiffusion</td>
    <td>49,357</td>
    <td></td>
  </tr>

  <tr>
    <td>CAF Dataset</td>
    <td>Pan-cancer CAF atlas</td>
    <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE210347">GSE210347</a></td>
    <td>73,402</td>
    <td>CAF_myo<br>CAF_infla<br>Fibroblasts<br>CAF_adi<br>CAF_EndoMT<br>CAF_PN<br>CAF_api</td>
  </tr>
  </tbody>
</table>

## Preprocessing

The five training sources are first restricted to their shared genes and combined, after which quality control is applied once to the integrated reference. The five standard-QC test datasets and the two CAF splits are processed separately using the same sequence: cells must contain at least 200 detected genes, genes must be detected in at least three cells, observation and gene names are made unique, and genes whose symbols begin with `RP` are removed. After recalculating quality-control metrics, cells are retained only when both total counts and the number of detected genes fall strictly between the dataset-specific 2nd and 98th percentiles and the mitochondrial fraction is below 10%.

After standard quality control, Scrublet is run on the integrated training reference and the five standard-QC test datasets with an expected doublet rate of 0.06. Predicted doublets are removed from the integrated training reference, whereas test-set doublet scores and predictions are retained without removing cells. Scrublet is not applied to the CAF data.

GSE135893 (`lung_full`) and GSE201333 (`sapiens_full`) follow the scDiffusion source-filtering policy instead: cells must contain at least 10 detected genes and genes must be detected in at least three cells. These two datasets do not undergo `RP`-gene removal, percentile filtering, mitochondrial-threshold filtering, or Scrublet. GSE135893 is reconstructed from the original-study count matrix, whereas GSE201333 uses the explicitly stored raw-count layer from the scDiffusion Figshare release.

For all final outputs, source counts are preserved before library-size normalization to 10,000 counts per cell followed by `log1p` transformation.

## Label Mapping

All cell-type labels used for scLightGAT training and testing are mapped from published source annotations; no model-predicted labels are used as ground truth. The table below lists the metadata fields used to map cell types and subtypes. For some datasets, a fallback field is used when the primary field is missing or cannot be mapped; all listed fields come from the same published annotation source.

<table>
<thead>
  <tr>
    <th>Dataset Role</th>
    <th>Source / Accession</th>
    <th>Cell-type Source</th>
    <th>Subtype Source</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="5">Training Datasets</td>
    <td>GSE131907</td>
    <td>GEO: <code>obs["Cell_subtype"]</code> (fallback: <code>obs["Cell_type.refined"]</code> / <code>obs["Cell_type"]</code>)</td>
    <td>GEO: <code>obs["Cell_subtype"]</code></td>
  </tr>
  <tr>
    <td>GSE176078</td>
    <td>GEO: <code>obs["celltype_subset"]</code> (fallback: <code>obs["celltype_minor"]</code>)</td>
    <td>GEO: <code>obs["celltype_subset"]</code></td>
  </tr>
  <tr>
    <td>GSE140228</td>
    <td>GEO: <code>obs["celltype_sub"]</code></td>
    <td>GEO: <code>obs["celltype_sub"]</code></td>
  </tr>
  <tr>
    <td>GSE132465</td>
    <td><a href="https://doi.org/10.5281/zenodo.7419236">scATOMIC</a>: <code>obs["harmonized_original_annotation"]</code> (fallback: <code>obs["original_annotation.1"]</code>)</td>
    <td><a href="https://doi.org/10.5281/zenodo.7419236">scATOMIC</a>: <code>obs["original_annotation.2"]</code></td>
  </tr>
  <tr>
    <td>GSE141445</td>
    <td><a href="https://doi.org/10.5281/zenodo.7419236">scATOMIC</a>: <code>obs["harmonized_original_annotation"]</code> (fallback: <code>obs["original_annotation.1"]</code>)</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="7">Test Datasets</td>
    <td>GSE153935</td>
    <td><a href="https://doi.org/10.5281/zenodo.7400873">Hanley et al.</a>: <code>obs["Cell.type2"]</code> (fallback: <code>obs["Lineage"]</code>)</td>
    <td></td>
  </tr>
  <tr>
    <td>GSE115978</td>
    <td>GEO: <code>obs["cell.types"]</code></td>
    <td></td>
  </tr>
  <tr>
    <td>GSE123139</td>
    <td><a href="https://li-melanoma-2018.s3.eu-west-1.amazonaws.com/Li2018_umi_counts.tar.gz">Li2018</a>: <code>obs["t_nk_mc_group"]</code> with <code>obs["Cell.type"]</code> for shared-naive T cells; <code>obs["non_t_nk_mc_group"]</code> for non-T/NK cells</td>
    <td></td>
  </tr>
  <tr>
    <td>GSE166555</td>
    <td>GEO: <code>obs["main_cell_type"]</code> with <code>obs["cell_type_imm_simple"]</code> / <code>obs["cell_type_str_simple"]</code></td>
    <td></td>
  </tr>
  <tr>
    <td>Zheng</td>
    <td><a href="https://doi.org/10.5281/zenodo.3357167">Abdelaal</a>: <code>Labels.csv</code></td>
    <td></td>
  </tr>
  <tr>
    <td>GSE135893</td>
    <td>GEO: <code>obs["celltype"]</code></td>
    <td></td>
  </tr>
  <tr>
    <td>GSE201333</td>
    <td><a href="https://doi.org/10.6084/m9.figshare.25335436.v2">scDiffusion</a>: <code>obs["free_annotation"]</code></td>
    <td></td>
  </tr>
  <tr>
    <td>CAF Dataset</td>
    <td>GSE210347</td>
    <td>GEO: <code>obs["celltype"]</code> (fibroblast/CAF selection)</td>
    <td>GEO: <code>obs["cluster"]</code> (C1&ndash;C8 mapping)</td>
  </tr>
</tbody>
</table>

The scLightGAT training labels are harmonized to the cell-type hierarchy shown below. For test datasets, annotations are mapped to the most specific supported class within the broad cell-type ontology; a supported coarse source annotation is retained rather than assigned to a more specific class without supporting evidence. For example, `T.cells` remains at that level rather than being mapped to `CD4+T cells` or `CD8+T cells`.

<table>
  <thead>
    <tr>
      <th>Major Type</th>
      <th>Subtype</th>
      <th>Major Type</th>
      <th>Subtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5">CD4+T cells</td>
      <td>CD4+Tfh/Th cells</td>
      <td rowspan="4">Plasma cells</td>
      <td>IgA+ Plasma</td>
    </tr>
    <tr>
      <td>CD4+exhausted T cells</td>
      <td>IgG+ Plasma</td>
    </tr>
    <tr>
      <td>CD4+memory T cells</td>
      <td>Plasma cells</td>
    </tr>
    <tr>
      <td>CD4+naive T cells</td>
      <td>Plasmablasts</td>
    </tr>
    <tr>
      <td>CD4+reg T cells</td>
      <td rowspan="2">DC</td>
      <td>DC</td>
    </tr>
    <tr>
      <td rowspan="4">CD8+T cells</td>
      <td>CD8+MAIT T cells</td>
      <td>cDC</td>
    </tr>
    <tr>
      <td>CD8+Naive T cells</td>
      <td>NK cells</td>
      <td></td>
    </tr>
    <tr>
      <td>CD8+exhausted T cells</td>
      <td>pDC</td>
      <td></td>
    </tr>
    <tr>
      <td>CD8+memory T cells</td>
      <td>Macrophages</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="5">B cells</td>
      <td>Follicular B cells</td>
      <td>Monocytes</td>
      <td></td>
    </tr>
    <tr>
      <td>Germinal B cells</td>
      <td>Epithelial cells</td>
      <td></td>
    </tr>
    <tr>
      <td>MALT B cells</td>
      <td>Endothelial cells</td>
      <td></td>
    </tr>
    <tr>
      <td>Memory B cells</td>
      <td>Fibroblasts</td>
      <td></td>
    </tr>
    <tr>
      <td>Naive B cells</td>
      <td>Mast cells</td>
      <td></td>
    </tr>
  </tbody>
</table>

### CAF Target Mapping

The GSE210347 source clusters are mapped to the seven CAF targets as follows:

| Source Cluster | Output Target |
| --- | --- |
| `C1` | `CAF_myo` |
| `C2` | `CAF_infla` |
| `C3` | `Fibroblasts` |
| `C4` | `CAF_adi` |
| `C5` | `Fibroblasts` |
| `C6` | `CAF_EndoMT` |
| `C7` | `CAF_PN` |
| `C8` | `CAF_api` |

## Installation

The scLightGAT preprocessing pipeline is implemented in Python. Clone the repository and create the Conda environment as follows:

```bash
git clone https://github.com/chenh2lab/scLightGAT
cd scLightGAT/preprocessing
conda create --name scLightGAT_data \
  --override-channels \
  --strict-channel-priority \
  --channel conda-forge \
  --channel bioconda \
  --file requirements.txt
conda activate scLightGAT_data
```

## Usage

Run the pipeline from the `preprocessing` directory to download the required public data, preprocess the datasets, harmonize the annotations, and generate all model-ready `.h5ad` files:

```bash
python3 main.py
```

This is equivalent to `python3 main.py build`. By default, the pipeline processes all datasets, uses seed `0` for Scrublet, stores checkpoints in `./.cache`, and writes the outputs to `./scLightGAT_data`. Downloaded source files are removed after a successful full build; run `python3 main.py build --keep-raw` to retain them.

Outputs are organized under `./scLightGAT_data/Integrated_training`, `./scLightGAT_data/Independent_testing`, `./scLightGAT_data/Subtype`, and `./scLightGAT_data/caf.data`.

Subtype datasets use the fixed donor-held-out assignments in [`subtype_donor_split.json`](subtype_donor_split.json), while the CAF dataset uses the fixed tissue-held-out split defined in [`utils/config.py`](utils/config.py).

## Output Format

The generated AnnData files expose the following core fields:

| AnnData Field | Contents |
| --- | --- |
| `.X` | Library-size-normalized and `log1p`-transformed expression |
| `.layers["counts"]` / `.layers["raw_counts"]` | Preserved pre-normalization source count matrices |
| `.layers["log_transformed"]` | Copy of the normalized and `log1p`-transformed `.X` matrix |
| `.obs["Celltype_training"]` | Model-facing target label: a harmonized major or supported coarse cell type for broad datasets, or one of the seven CAF targets for CAF outputs |
| `.obs["Celltype_subtraining"]` | Source-backed subtype label; present only in the integrated-training and derived subtype outputs |
| `.obs["Data_source"]` | Source dataset identifier |
