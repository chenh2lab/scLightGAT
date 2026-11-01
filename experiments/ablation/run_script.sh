#!/usr/bin/env bash
set -eo pipefail

ablation_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$ablation_root/../.." && pwd)"
result_root="$ablation_root/result"
cache_root="$ablation_root/.cache"
python_executable=/Group16T/common/ccuc/miniconda3_gb10/envs/scLightGAT/bin/python3
arguments=("$@")
save_path=""
help_requested=0

mkdir -p "$cache_root/matplotlib" "$cache_root/numba"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cache_root/matplotlib"
export NUMBA_CACHE_DIR="$cache_root/numba"
export SCIPY_ARRAY_API=1

for ((index = 0; index < ${#arguments[@]}; index++)); do
    case "${arguments[$index]}" in
        --save_path)
            if ((index + 1 >= ${#arguments[@]})); then
                echo "Error: --save_path requires a value." >&2
                exit 2
            fi
            save_path="${arguments[$((index + 1))]}"
            save_index=$((index + 1))
            index=$((index + 1))
            ;;
        --save_path=*)
            save_path="${arguments[$index]#--save_path=}"
            save_index=$index
            ;;
        -h|--help)
            help_requested=1
            ;;
    esac
done

if ((help_requested == 0)); then
    if [[ -z "$save_path" ]]; then
        echo "Error: --save_path is required." >&2
        exit 2
    fi
    if [[ "$save_path" != /* ]]; then
        save_path="$project_root/$save_path"
    fi
    resolved_result_root="$(realpath -m "$result_root")"
    resolved_save_path="$(realpath -m "$save_path")"
    case "$resolved_save_path" in
        "$resolved_result_root"|"$resolved_result_root"/*) ;;
        *)
            echo "Error: ablation results must stay under $resolved_result_root" >&2
            exit 2
            ;;
    esac
    mkdir -p "$resolved_save_path"
    if [[ "${arguments[$save_index]}" == --save_path=* ]]; then
        arguments[$save_index]="--save_path=$resolved_save_path"
    else
        arguments[$save_index]="$resolved_save_path"
    fi
fi

cd "$project_root"

for variant in No_CDVAE No_GAT Only_LightGBM; do
    hparam_path="$ablation_root/$variant/scLightGAT/hparam.py"
    cpu_workers="$(
        "$python_executable" -c \
            'import runpy, sys; print(runpy.run_path(sys.argv[1])["CPU_WORKERS"])' \
            "$hparam_path"
    )"
    if ! [[ "$cpu_workers" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: CPU_WORKERS in $hparam_path must be a positive integer." >&2
        exit 2
    fi
    export OMP_NUM_THREADS="$cpu_workers"
    export OPENBLAS_NUM_THREADS="$cpu_workers"
    export MKL_NUM_THREADS="$cpu_workers"
    export BLIS_NUM_THREADS="$cpu_workers"
    export VECLIB_MAXIMUM_THREADS="$cpu_workers"
    export NUMEXPR_NUM_THREADS="$cpu_workers"
    export NUMBA_NUM_THREADS="$cpu_workers"

    echo "Running ${variant}"
    "$python_executable" "$ablation_root/$variant/main.py" "${arguments[@]}"
done
