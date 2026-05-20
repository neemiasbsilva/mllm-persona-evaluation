#!/usr/bin/env bash
# Re-run every analysis notebook end-to-end and write the executed copies
# back to the source path (so figures and tables under `figures/` stay fresh).
#
# Usage:
#   bash scripts/run_all_notebooks.sh             # run every notebook
#   bash scripts/run_all_notebooks.sh 01 03       # run a subset (matches name)
#
# Requirements: `uv` and the project's lockfile.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NOTEBOOKS=(
    "notebooks/01_baseline_distribution_eda.ipynb"
    "notebooks/02_convergence_analysis_profile.ipynb"
    "notebooks/03_perceptsent_cv_agreement.ipynb"
    "notebooks/05_rq1_boxplots_cosine.ipynb"
)

TIMEOUT="${NB_TIMEOUT:-900}"

select_nb() {
    local nb="$1"; shift
    [[ "$#" -eq 0 ]] && return 0
    for filter in "$@"; do
        [[ "$nb" == *"$filter"* ]] && return 0
    done
    return 1
}

failed=()
for nb in "${NOTEBOOKS[@]}"; do
    if ! select_nb "$nb" "$@"; then
        continue
    fi
    echo "── Executing: $nb ─────────────────────────────────────────────"
    if uv run jupyter nbconvert \
            --to notebook --execute \
            --ExecutePreprocessor.timeout="$TIMEOUT" \
            --output "$(basename "$nb")" \
            --output-dir "$(dirname "$nb")" \
            "$nb"; then
        echo "✓ $nb"
    else
        echo "✗ $nb"
        failed+=("$nb")
    fi
done

if [[ "${#failed[@]}" -gt 0 ]]; then
    echo
    echo "Failed notebooks:"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi

echo
echo "All notebooks executed successfully. Figures written to figures/."
