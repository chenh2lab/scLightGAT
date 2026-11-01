#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /Group16T/common/ccuc/miniconda3_gb10/etc/profile.d/conda.sh

train_path=""
test_path=""
save_path=""
train_celltype="celltype"
test_celltype="celltype"
train_batch="batch"
test_batch="batch"
test_batch_map=""
n_runs=5
n_jobs=12
scgpt_model_dir="scGPT/models/scGPT_human"

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

export OMP_NUM_THREADS="$n_jobs"
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS="$n_jobs"
export MKL_NUM_THREADS="$n_jobs"
export MKL_DYNAMIC=FALSE
export BLIS_NUM_THREADS="$n_jobs"
export VECLIB_MAXIMUM_THREADS="$n_jobs"
export NUMEXPR_NUM_THREADS="$n_jobs"
export RCPP_PARALLEL_NUM_THREADS="$n_jobs"
export PYTHONDONTWRITEBYTECODE=1
export SCIPY_ARRAY_API=1

echo "Train dataset: $train_path"
echo "Test dataset: $test_path"
echo "Save path: $save_path"
echo "Train celltype: $train_celltype"
echo "Test celltype: $test_celltype"
echo "Train batch: $train_batch"
echo "Test batch: $test_batch"
echo "Test batch map: $test_batch_map"

conda activate sctools
echo "Running Celltypist cell type annotation"
python3 Celltypist/Celltypist.py \
    --train_path "$train_path" \
    --test_path "$test_path" \
    --save_path "$save_path" \
    --train_celltype "$train_celltype" \
    --test_celltype "$test_celltype" \
    --train_batch "$train_batch" \
    --test_batch "$test_batch" \
    --test_batch_map "$test_batch_map" \
    --n_jobs "$n_jobs" \
    --run_times "$n_runs"

conda activate CHETAH
echo "Running CHETAH cell type annotation"
python3 CHETAH/CHETAH.py \
    --train_path "$train_path" \
    --test_path "$test_path" \
    --save_path "$save_path" \
    --train_celltype "$train_celltype" \
    --test_celltype "$test_celltype" \
    --train_batch "$train_batch" \
    --test_batch "$test_batch" \
    --test_batch_map "$test_batch_map" \
    --n_jobs "$n_jobs" \
    --run_times "$n_runs"

conda activate Seurat
echo "Running Seurat cell type annotation"
python3 Seurat/Seurat.py \
    --train_path "$train_path" \
    --test_path "$test_path" \
    --save_path "$save_path" \
    --train_celltype "$train_celltype" \
    --test_celltype "$test_celltype" \
    --train_batch "$train_batch" \
    --test_batch "$test_batch" \
    --test_batch_map "$test_batch_map" \
    --n_jobs "$n_jobs" \
    --run_times "$n_runs"

conda activate SingleR
echo "Running SingleR cell type annotation"
python3 SingleR/SingleR.py \
    --train_path "$train_path" \
    --test_path "$test_path" \
    --save_path "$save_path" \
    --train_celltype "$train_celltype" \
    --test_celltype "$test_celltype" \
    --train_batch "$train_batch" \
    --test_batch "$test_batch" \
    --test_batch_map "$test_batch_map" \
    --n_jobs "$n_jobs" \
    --run_times "$n_runs"

conda activate scGPT
echo "Validating scGPT checkpoint"
python3 scGPT/download_checkpoint.py \
    --output_dir "$scgpt_model_dir"
echo "Running scGPT cell type annotation"
python3 scGPT/scGPT.py \
    --train_path "$train_path" \
    --test_path "$test_path" \
    --save_path "$save_path" \
    --train_celltype "$train_celltype" \
    --test_celltype "$test_celltype" \
    --train_batch "$train_batch" \
    --test_batch "$test_batch" \
    --test_batch_map "$test_batch_map" \
    --n_jobs "$n_jobs" \
    --run_times "$n_runs" \
    --scgpt_model_dir "$scgpt_model_dir"

conda activate sctools
echo "Running scANVI cell type annotation"
python3 scANVI/scANVI.py \
    --train_path "$train_path" \
    --test_path "$test_path" \
    --save_path "$save_path" \
    --train_celltype "$train_celltype" \
    --test_celltype "$test_celltype" \
    --train_batch "$train_batch" \
    --test_batch "$test_batch" \
    --test_batch_map "$test_batch_map" \
    --n_jobs "$n_jobs" \
    --run_times "$n_runs"

conda activate scGraphformer
echo "Running scGraphformer cell type annotation"
python3 scGraphformer/scGraphformer.py \
    --train_path "$train_path" \
    --test_path "$test_path" \
    --save_path "$save_path" \
    --train_celltype "$train_celltype" \
    --test_celltype "$test_celltype" \
    --train_batch "$train_batch" \
    --test_batch "$test_batch" \
    --test_batch_map "$test_batch_map" \
    --n_jobs "$n_jobs" \
    --run_times "$n_runs"
