#!/usr/bin/env bash
#
# Runs the full experiment campaign behind the paper: 8 experiments x 25 seeds x
# 4 strategies, across both alloy systems.
#
#   ./replicate.sh                          # everything
#   ./replicate.sh --dry-run                # print the plan, run nothing
#   ./replicate.sh --only Magnets-MAIN      # one experiment
#   ./replicate.sh --only Magnets-MAIN --seeds 2489 9596
#
# Results land in results_magnets_v3/ and results_mpd_v3/, under
# <experiment>/seed_<N>/<strategy>/. Completed seed x strategy pairs are skipped,
# so an interrupted run can be restarted with the same command.
#
# Two of the four strategies (Adaptive-Beta, Rule-Swap) call an LLM once per BO
# iteration and need OPENAI_API_KEY set, in the environment or in .env. Those
# calls are not deterministic, so a rerun will not reproduce the published
# numbers exactly. Fixed-Mixed and qEHVI need no key and are reproducible from
# their seeds.
#
# Expect days of wall-clock time and real API spend for the whole campaign. Use
# --only and --seeds to run a slice.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ALL_EXPERIMENTS=(
    MP-D_MAIN  MP-D_TIME  MP-D_BUDGET  MP-D_DUAL
    Magnets-MAIN  Magnets-TIME  Magnets-BUDGET  Magnets-DUAL
)

PYTHON=python3
[ -x .venv/bin/python ] && PYTHON=.venv/bin/python

DRY_RUN=""
SEEDS=()
EXPERIMENTS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --only)    EXPERIMENTS+=("$2"); shift 2 ;;
        --seeds)   shift
                   while [ $# -gt 0 ] && [[ "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
        -h|--help) sed -n '2,25p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[ ${#EXPERIMENTS[@]} -eq 0 ] && EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")

if [ -z "$DRY_RUN" ] && [ -z "${OPENAI_API_KEY:-}" ] && ! grep -qs '^OPENAI_API_KEY=.' .env; then
    echo "OPENAI_API_KEY is not set and .env does not define it." >&2
    echo "Adaptive-Beta and Rule-Swap cannot run without it. See .env.example." >&2
    exit 1
fi

for exp in "${EXPERIMENTS[@]}"; do
    case "$exp" in
        MP-D_*)    output_dir=results_mpd_v3 ;;
        Magnets-*) output_dir=results_magnets_v3 ;;
        *) echo "unknown experiment: $exp" >&2; exit 2 ;;
    esac

    echo "=== $exp -> $output_dir ==="
    "$PYTHON" -m resource_allocation.experiments.run_campaign \
        --experiment "$exp" \
        --output-dir "$output_dir" \
        ${SEEDS[@]+--seeds "${SEEDS[@]}"} \
        $DRY_RUN

    if [ -z "$DRY_RUN" ]; then
        {
            echo "experiment:  $exp"
            echo "finished:    $(date -Is)"
            echo "git commit:  $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
            echo "host:        $(hostname)"
        } >> "$output_dir/_run_info.txt"
    fi
done

echo "Done. Results in results_magnets_v3/ and results_mpd_v3/."
