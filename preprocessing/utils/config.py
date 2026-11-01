from __future__ import annotations

TRAIN_DATASETS = [
    "GSE131907",
    "GSE176078",
    "GSE140228",
    "GSE132465",
    "GSE141445",
]

TEST_DATASETS = [
    "GSE153935",
    "GSE115978",
    "GSE123139",
    "GSE166555",
    "Zheng_sorted",
]

EXTRA_DATASETS = ["lung_full", "sapiens_full"]
CAF_DATASETS = ["CAF_train", "CAF_test"]

ALL_LOGICAL_DATASETS = (
    TRAIN_DATASETS + TEST_DATASETS + EXTRA_DATASETS + CAF_DATASETS
)

DATASET_GROUPS = {
    "main": TRAIN_DATASETS + TEST_DATASETS,
    "train": TRAIN_DATASETS,
    "test": TEST_DATASETS,
    "extra": EXTRA_DATASETS,
    "caf": CAF_DATASETS,
}


# Canonical broad cell-type ontology used by the scLightGAT thesis/code.
# No other label is allowed in Celltype_training for the main/extra datasets.
MAIN_MAJOR_LABELS = {
    "CD4+T cells",
    "CD8+T cells",
    "NK cells",
    "pDC",
    "B cells",
    "Plasma cells",
    "DC",
    "Macrophages",
    "Monocytes",
    "Epithelial cells",
    "Endothelial cells",
    "Fibroblasts",
    "Mast cells",
}

OUTPUT_PATHS = {
    "train": "Integrated_training/train.h5ad",
    "GSE153935": "Independent_testing/GSE153935.h5ad",
    "GSE115978": "Independent_testing/GSE115978.h5ad",
    "GSE123139": "Independent_testing/GSE123139.h5ad",
    "GSE166555": "Independent_testing/GSE166555.h5ad",
    "Zheng_sorted": "Independent_testing/Zhengsorted.h5ad",
    "lung_full": "Independent_testing/lung_full.h5ad",
    "sapiens_full": "Independent_testing/sapiens_full.h5ad",
    "CAF_train": "caf.data/caf_train.h5ad",
    "CAF_test": "caf.data/caf_test.h5ad",
}

GEO_PATTERNS = {
    "GSE131907": [
        r"raw_UMI_matrix\.txt\.gz$",
        r"cell_annotation\.txt\.gz$",
    ],
    "GSE176078": [r"Wu_etal_2021_BRCA_scRNASeq\.tar\.gz$"],
    "GSE140228": [
        r"UMI_counts_Droplet\.mtx\.gz$",
        r"Droplet_barcodes\.tsv\.gz$",
        r"Droplet_cellinfo\.tsv\.gz$",
        r"Droplet_genes\.tsv\.gz$",
        r"cell_info_Smartseq2\.tsv\.gz$",
        r"gene_info_Smartseq2\.tsv\.gz$",
        r"read_counts_Smartseq2\.csv\.gz$",
    ],
    "GSE132465": [
        r"raw_UMI_count_matrix\.txt\.gz$",
        r"cell_annotation\.txt\.gz$",
    ],
    "GSE141445": [r"GSE141445_RAW\.tar$"],
    "GSE153935": [r"TLDS_AllCells\.txt\.gz$"],
    "GSE115978": [r"counts\.csv\.gz$", r"cell\.annotations\.csv\.gz$"],
    "GSE123139": [r"GSE123139_RAW\.tar$"],
    "GSE166555": [r"GSE166555_RAW\.tar$", r"GSE166555_meta_data\.tsv\.gz$"],
    "lung_full": [
        r"GSE135893_matrix\.mtx\.gz$",
        r"GSE135893_barcodes\.tsv\.gz$",
        r"GSE135893_genes\.tsv\.gz$",
        r"GSE135893_IPF_metadata\.csv\.gz$",
    ],
    "CAF": [r"GSE210347_counts\.Rds\.gz$", r"GSE210347_meta\.txt\.gz$"],
}

SCATOMIC_ZENODO = "7419236"
GSE153935_ZENODO = "7400873"
ZHENG_ZENODO = "3357167"
LI2018_URL = (
    "https://li-melanoma-2018.s3.eu-west-1.amazonaws.com/"
    "Li2018_umi_counts.tar.gz"
)

# Official scDiffusion paper-data deposit.  The public project README points to
# this Figshare dataset for the h5ad files used in the paper.
SCDIFFUSION_FIGSHARE_ARTICLE = "25335436"
SCDIFFUSION_FIGSHARE_VERSION = "2"

SCDIFFUSION_SAPIENS_CELLTYPES = {
    "fibroblast",
    "macrophage",
    "memory b cell",
    "naive b cell",
    "nk cell",
    "plasma cell",
}

SCDIFFUSION_LUNG_CELLTYPES = {
    "AT1",
    "AT2",
    "Basal",
    "Ciliated",
    "Differentiating Ciliated",
    "KRT5-/KRT17+",
    "MUC5AC+ High",
    "MUC5B+",
    "SCGB3A2+",
    "SCGB3A2+ SCGB1A1+",
    "Proliferating Epithelial Cells",
    "Transitional AT2",
    "Mesothelial Cells",
    "Endothelial Cells",
    "Lymphatic Endothelial Cells",
    "Fibroblasts",
    "HAS1 High Fibroblasts",
    "Myofibroblasts",
    "PLIN2+ Fibroblasts",
    "Smooth Muscle Cells",
    "Macrophages",
    "Proliferating Macrophages",
    "Monocytes",
    "Mast Cells",
    "B Cells",
    "Plasma Cells",
    "NK Cells",
    "T Cells",
    "Proliferating T Cells",
    "cDCs",
    "pDCs",
}

CAF_TRAIN_TISSUES = {"Pancreas", "Bladder", "Ovary", "Colon"}
CAF_TEST_TISSUES = {
    "Stomach",
    "Lung",
    "Breast",
    "Thyroid",
    "Prostate",
    "Intrahepatic",
    "Liver",
}

# CAF is a separate seven-class classification task.
CAF_TARGET_LABELS = {
    "CAF_EndoMT",
    "CAF_adi",
    "CAF_api",
    "CAF_myo",
    "CAF_PN",
    "CAF_infla",
    "Fibroblasts",
}
