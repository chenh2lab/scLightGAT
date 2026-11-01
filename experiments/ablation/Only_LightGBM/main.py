"""Entry point for the DGE-only LightGBM ablation."""

import os
import sys
from pathlib import Path


VARIANT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = VARIANT_ROOT.parents[2]
CACHE_ROOT = VARIANT_ROOT.parent / ".cache"
MATPLOTLIB_CACHE = CACHE_ROOT / "matplotlib"
NUMBA_CACHE = CACHE_ROOT / "numba"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
NUMBA_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE))
os.environ.setdefault("SCIPY_ARRAY_API", "1")
for import_root in (str(VARIANT_ROOT), str(PROJECT_ROOT)):
    while import_root in sys.path:
        sys.path.remove(import_root)
sys.path.insert(0, str(VARIANT_ROOT))
sys.path.insert(1, str(PROJECT_ROOT))

from scLightGAT.hparam import CPU_WORKERS


for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[variable] = str(CPU_WORKERS)

from scLightGAT import scLightGAT
from utils.pipeline import create_argument_parser, run_annotation_pipeline


TOOL_NAME = "Only_LightGBM"
PREDICTION_KEY = "scLightGAT_pred"


def _model_dir(run_out_path):
    return Path(run_out_path) / "_model"


def _drop_unused_expression_copies(adata):
    adata.raw = None
    for layer in list(adata.layers):
        if layer not in {"counts", "log_transformed"}:
            del adata.layers[layer]
    return adata


def train_sclightgat(adata, seed, run_out_path, args):
    model = scLightGAT(seed=seed)
    model.fit_reference(
        _drop_unused_expression_copies(adata),
        label_key=args.train_celltype,
        batch_key=args.train_batch,
    )
    model.save(_model_dir(run_out_path))
    return model


def predict_sclightgat(adata, model, pred_key, seed, run_out_path, args):
    del seed, run_out_path
    return model.predict_query(
        _drop_unused_expression_copies(adata),
        batch_key=args.test_batch,
        pred_key=pred_key,
    )


def load_sclightgat_model(run_out_path, seed=None, args=None):
    del seed, args
    return scLightGAT.load(_model_dir(run_out_path))


def sclightgat_model_is_complete(run_out_path, seed=None, args=None):
    del seed, args
    return scLightGAT.is_complete(_model_dir(run_out_path))


def create_parser():
    parser = create_argument_parser(TOOL_NAME)
    parser.set_defaults(
        train_celltype="Celltype_training",
        test_celltype="Celltype_training",
        train_batch="Data_source",
        test_batch=None,
    )
    return parser


def main(argv=None):
    args = create_parser().parse_args(argv)
    run_annotation_pipeline(
        TOOL_NAME,
        PREDICTION_KEY,
        train_sclightgat,
        predict_sclightgat,
        args,
        load_model_fn=load_sclightgat_model,
        model_is_complete_fn=sclightgat_model_is_complete,
        expression_layers=("counts", "log_transformed"),
        share_train_expression=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
