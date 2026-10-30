#!/bin/sh
# run_final.sh -- the final scoring, summary and analysis, entirely inside final/.
#   ./run_final.sh svcomp | tdx | all
# Run from the directory that holds the scripts and datasets/ (~/EvalBMC).
# Every CBMC run already done is reused from final/cache/verify.sqlite.
set -e
F=final
WHAT=${1:-all}
S="python3 score_predictions.py --unsafe-rule exposure --jobs 6 --verify-db $F/cache/verify.sqlite"

svcomp() {
  D=$F/svcomp; J=$F/inputs/svcomp_loops.json; P=datasets/cleaned/svcomp_clean
  mkdir -p $D/scores $D/analysis $D/logs
  $S --pred $D/predictions/qwen3.5.jsonl   --kstar $D/kstar.csv --json $J --dataset $P --out $D/scores/qwen3.5.csv   > $D/logs/score_qwen3.5.log 2>&1
  $S --pred $D/predictions/syntactic.jsonl --kstar $D/kstar.csv --json $J --dataset $P --out $D/scores/syntactic.csv > $D/logs/score_syntactic.log 2>&1
  python3 results_summary.py --kstar $D/kstar.csv \
      --score "Syntactic=$D/scores/syntactic.csv" --score "Qwen3.5=$D/scores/qwen3.5.csv" > $D/results_summary.txt
  python3 analysis.py --json $J --kstar $D/kstar.csv \
      --score "Syntactic=$D/scores/syntactic.csv" --score "Qwen3.5=$D/scores/qwen3.5.csv" \
      --out-dir $D/analysis > $D/analysis/summary.txt
}

tdx() {
  D=$F/tdx; J=$F/inputs/tdx_loops.json; P=datasets/tdx_cprover_default
  mkdir -p $D/scores $D/analysis $D/logs
  $S --pred $D/predictions/syntactic.jsonl --kstar $D/kstar.csv --json $J --dataset $P --out $D/scores/syntactic.csv > $D/logs/score_syntactic.log 2>&1
  $S --pred $D/predictions/qwen3.5.jsonl   --kstar $D/kstar.csv --json $J --dataset $P --out $D/scores/qwen3.5.csv   > $D/logs/score_qwen3.5.log 2>&1
  python3 results_summary.py --kstar $D/kstar.csv \
      --score "Syntactic=$D/scores/syntactic.csv" --score "Qwen3.5=$D/scores/qwen3.5.csv" > $D/results_summary.txt
  python3 analysis.py --json $J --kstar $D/kstar.csv \
      --score "Syntactic=$D/scores/syntactic.csv" --score "Qwen3.5=$D/scores/qwen3.5.csv" \
      --group-by-task-name --out-dir $D/analysis > $D/analysis/summary.txt
}

case "$WHAT" in
  svcomp) svcomp ;;
  tdx)    tdx ;;
  all)    svcomp; tdx ;;
  *) echo "usage: $0 svcomp|tdx|all"; exit 2 ;;
esac
echo "done: $WHAT"