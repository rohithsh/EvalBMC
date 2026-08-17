# Evaluating Approaches for Determining Unrolling Bounds in BMC

The scripts run in three phases: build the dataset, derive the ground truth,
then evaluate strategies against it.


## Requirements

```bash
# CBMC 6.8 or later
cbmc --version

# Python
pip install pyyaml libclang requests python-dotenv
```

Add the API keys for the LLMs in a `.env` file:

```
API_KEY=primary-key
API_KEY_BACKUP=second-key       # optional; used when the primary is rate limited
```

## Getting the benchmarks

```bash
cd datasets
git clone --depth 1 https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks.git
cd ..
```

## Phase 1: build the dataset

```bash
python3 svcomp_preprocessor.py \
  --root  datasets/sv-benchmarks/c \
  --sets  Loops.set Arrays.set Heap.set LinkedLists.set ControlFlow.set BitVectors.set \
  --dest  datasets/cleaned \
  --scripts . --jobs 8
```

## Phase 2: derive the ground truth

```bash
python3 derive_kstar_all.py \
  --json      datasets/cleaned/loops.json \
  --dataset   datasets/cleaned/svcomp_clean \
  --out       results/kstar_all.csv \
  --trace-dir results/kstar_traces \
  --jobs 4 --mode both \
  --budget 600 --per-k-timeout 180 --max-k 100000
```

This takes **1–2 days** at `--jobs 16`.

Deeper analysis of the finished run:

```bash
python3 analyse_kstar_all.py --csv results/kstar_all.csv
```

## Phase 3: evaluate the strategies

### 3.1 Create an evaluation sample (Optional)

Every strategy is measured on the same fixed set, drawn once:

```bash
python3 build_eval_sample.py \
  --kstar results/kstar_all.csv \
  --out   results/eval_sample.csv \
  --n 200 --n-timeout 40 --max-per-dir 15 --seed 0
```

### 3.2 Run a strategy
**LLM**, with the verdict withheld (blind prediction):

```bash
python3 predict_bounds.py \
  --json    datasets/cleaned/loops.json \
  --dataset datasets/cleaned/svcomp_clean \
  --out     results/pred_llm_blind.jsonl \
  --provider openai --base-url <url> \
  --model <model> --mode full --max-tokens 4000 \
  --only-tasks results/eval_sample.csv --jobs 4
```

Inspect prompts (on a sample program):

```bash
python3 build_prompts.py --json datasets/cleaned/loops.json \
  --dataset datasets/cleaned/svcomp_clean \
  --print loops/bubble_sort-1.yml --property reach
```

### 3.3 Score

```bash
python3 score_predictions.py \
  --pred    results/pred_llm_blind.jsonl \
  --kstar   results/kstar_all.csv \
  --json    datasets/cleaned/loops.json \
  --dataset datasets/cleaned/svcomp_clean \
  --out     results/scores_llm_blind.csv \
  --jobs 4 --margins 1.5 2 4
```