#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${CONDA_BASE:-}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "Error: conda is not available in PATH." >&2
        exit 1
    fi
    CONDA_BASE="$(conda info --base)"
fi

CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"

if [[ ! -f "$CONDA_SH" ]]; then
    echo "Error: Conda initialization script not found: $CONDA_SH" >&2
    exit 1
fi

source "$CONDA_SH"
conda activate "${SCLIGHTGAT_CONDA_ENV:-scLightGAT}"

cd "$SCRIPT_DIR"

cache_dir="$SCRIPT_DIR/.cache"
mkdir -p "$cache_dir/matplotlib" "$cache_dir/numba"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cache_dir/matplotlib"
export NUMBA_CACHE_DIR="$cache_dir/numba"
export SCIPY_ARRAY_API=1

cpu_workers="$(
    python3 -c \
        'import runpy, sys; print(runpy.run_path(sys.argv[1])["CPU_WORKERS"])' \
        "$SCRIPT_DIR/scLightGAT/scLightGAT/hparam.py"
)"
if ! [[ "$cpu_workers" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: CPU_WORKERS in hparam.py must be a positive integer." >&2
    exit 2
fi
export OMP_NUM_THREADS="$cpu_workers"
export OPENBLAS_NUM_THREADS="$cpu_workers"
export MKL_NUM_THREADS="$cpu_workers"
export BLIS_NUM_THREADS="$cpu_workers"
export VECLIB_MAXIMUM_THREADS="$cpu_workers"
export NUMEXPR_NUM_THREADS="$cpu_workers"
export NUMBA_NUM_THREADS="$cpu_workers"

train_path=""
test_path=""
save_path=""
train_celltype="Celltype_training"
test_celltype="Celltype_training"
train_batch="Data_source"
test_batch="None"
test_batch_map=""
n_runs=1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --train_path)
            train_path="$2"
            shift 2
            ;;
        --test_path)
            test_path="$2"
            shift 2
            ;;
        --save_path)
            save_path="$2"
            shift 2
            ;;
        --train_celltype)
            train_celltype="$2"
            shift 2
            ;;
        --test_celltype)
            test_celltype="$2"
            shift 2
            ;;
        --train_batch)
            train_batch="$2"
            shift 2
            ;;
        --test_batch)
            test_batch="$2"
            shift 2
            ;;
        --test_batch_map)
            test_batch_map="$2"
            shift 2
            ;;
        --n_runs)
            n_runs="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

if [ -z "$train_path" ]; then
    echo "Error: --train_path is required."
    exit 1
fi

if [ -z "$test_path" ]; then
    echo "Error: --test_path is required."
    exit 1
fi

if [ -z "$save_path" ]; then
    echo "Error: --save_path is required."
    exit 1
fi

echo "Train dataset: $train_path"
echo "Test dataset: $test_path"
echo "Save path: $save_path"
echo "Train celltype: $train_celltype"
echo "Test celltype: $test_celltype"
echo "Train batch: $train_batch"
echo "Test batch: $test_batch"
echo "Test batch map: $test_batch_map"

echo "Running scLightGAT cell type annotation"
python3 scLightGAT/main.py \
    --train_path "$train_path" \
    --test_path "$test_path" \
    --save_path "$save_path" \
    --train_celltype "$train_celltype" \
    --test_celltype "$test_celltype" \
    --train_batch "$train_batch" \
    --test_batch "$test_batch" \
    --test_batch_map "$test_batch_map" \
    --n_runs "$n_runs"
