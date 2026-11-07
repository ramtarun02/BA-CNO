#!/usr/bin/env bash
# Run the CylinderFlow ablation ladder on a single Linux GPU box (RunPod etc.).
# All configs train concurrently on one GPU: the graphs are small, so a single
# run leaves the GPU mostly idle. Every config still gets the same step budget.
#
#   bash scripts/cylinder/run_pod.sh setup                 # download + convert to $DATA (once per volume)
#   bash scripts/cylinder/run_pod.sh setup-lite            # valid, test + 50 train trajectories (timing only)
#   bash scripts/cylinder/run_pod.sh timing [steps]        # concurrent timing run -> $RUNS/timing/budget.json
#   bash scripts/cylinder/run_pod.sh train <max_steps>     # launch all configs in the background (resumes)
#   bash scripts/cylinder/run_pod.sh status                # last log line of each run
#
# Data and checkpoints live on the persistent volume (/workspace by default),
# so a restarted pod resumes from <run>/last.pt by calling `train` again.
set -euo pipefail
ROOT=${ROOT:-/workspace}
DATA=${DATA:-$ROOT/data/cylinder_flow}
RUNS=${RUNS:-$ROOT/outputs/cylinder}
CONFIGS=${CONFIGS:-"mgn ba_cno_soft ba_cno_hard ba_cno_none_fluxdec ba_cno_hard_fluxdec"}
# one CPU thread per process: N concurrent runs otherwise oversubscribe the cores
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

launch() {  # launch <tag-dir> <max_steps> <log_every>
  local base=$1 steps=$2 log_every=$3
  for c in $CONFIGS; do
    mkdir -p "$base/$c"
    nohup python scripts/cylinder/train.py --config "configs/cylinder/$c.yaml" --override \
      output_dir="$base/$c" data.root="$DATA" train.max_steps="$steps" train.log_every="$log_every" \
      >> "$base/$c/train.log" 2>&1 &
    echo "started $c (pid $!) -> $base/$c/train.log"
  done
}

case ${1:-} in
  setup)
    pip install -q -e .
    for split in valid test train; do
      if [ ! -d "$DATA/$split" ]; then
        bash scripts/cylinder/download.sh "$DATA/raw" "$split"
        python scripts/cylinder/convert.py --raw "$DATA/raw" --out "$DATA" --splits "$split"
        rm -f "$DATA/raw/$split.tfrecord"
      fi
    done
    echo "train trajectories: $(ls "$DATA/train" | wc -l)"
    ;;
  setup-lite)  # enough for timing: it/s does not depend on how many trajectories exist
    pip install -q -e .
    bash scripts/cylinder/download.sh "$DATA/raw" valid test train:50
    python scripts/cylinder/convert.py --raw "$DATA/raw" --out "$DATA" --splits valid test
    python scripts/cylinder/convert.py --raw "$DATA/raw" --out "$DATA" --splits train --max 50
    rm -f "$DATA"/raw/*.tfrecord
    echo "train trajectories: $(ls "$DATA/train" | wc -l) (lite; run 'setup' after removing $DATA/train for the full set)"
    ;;
  timing)
    steps=${2:-2000}
    rm -rf "$RUNS/timing"
    launch "$RUNS/timing" "$steps" 500
    wait
    python scripts/cylinder/budget.py --runs "$RUNS"/timing/* --hours-per-config "${HOURS:-36}" --out "$RUNS/timing/budget.json"
    ;;
  train)
    steps=${2:?usage: run_pod.sh train <max_steps>}
    launch "$RUNS" "$steps" 1000
    echo "follow with: bash scripts/cylinder/run_pod.sh status"
    ;;
  status)
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    for c in $CONFIGS; do
      printf '%-22s %s\n' "$c" "$(grep -E '^\{"step"|Resumed|done:|Error' "$RUNS/$c/train.log" 2>/dev/null | tail -1)"
    done
    ;;
  *)
    sed -n '2,14p' "$0"; exit 1 ;;
esac
