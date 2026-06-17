# A Graph-Theoretical Framework for Analyzing the Behavior of Causal Language Models

This repository implements the graph-theoretical evaluation framework from [*A Graph-Theoretical
Framework for Analyzing the Behavior of Causal Language Models*](https://aclanthology.org/2025.emnlp-main.1014/)
(Rahnamoun & Shamsfard, EMNLP 2025). It measures structural properties (spectral entropy, graph
energy, density) of the token-transition graph a causal language model produces while generating
text.

The repository has two parts:

- **`graph_metrics_eval.py`** (this folder) — a general-purpose, fully parameterized CLI tool
  built on top of the paper's method, documented below.
- **`src/main_paper/`** — the original scripts used to produce the paper's results, kept as-is
  for reproducibility, described in the [Paper Reproduction](#paper-reproduction-srcmain_paper)
  section.

## Paper Reproduction (`src/main_paper`)

```
src/main_paper/
├── dataset_sampler.py
├── multi_model_eval.py
├── training_dynamics.py
├── sampled_datasets/
└── results/
```

### `sampled_datasets/`

The fixed prompt samples drawn from each benchmark dataset that the paper's experiments were
run on. `training_dynamics.py` and `multi_model_eval.py` both read from these files so that every
checkpoint and every model is evaluated on the prompts, making the metrics directly
comparable across runs.

### `dataset_sampler.py`

Draws and saves the fixed prompt samples into `sampled_datasets/` so the rest of the pipeline can
reuse them.

### `training_dynamics.py`

Tracks how the graph metrics evolve **across training checkpoints of a single Pythia model**
(`EleutherAI/pythia-410m-deduped`, checkpoints `step3000` through `step143000`). For each
checkpoint it: loads the model and tokenizer at that revision; for every sampled prompt, generates
new tokens and builds a directed graph where nodes are tokens and edges connect each token to the
top-60 candidate next-tokens, weighted by their generation probability; computes spectral entropy,
 graph energy, and density on that
graph; averages each metric across all sampled prompts for that checkpoint; and appends the result
to `metrics_per_step.json`. After all checkpoints are processed, it produces three sets of plots:
one line chart per metric across training steps, a combined multi-panel comparison figure, and a
single chart with all metrics normalized and overlaid.

### `multi_model_eval.py`

Runs the same graph-construction and metric pipeline, but **compares 13 different pretrained
models** instead of checkpoints of one model (OPT 125M/350M/1.3B, Pythia
14M/31M/70M/160M/410M/1B/1.4B, BLOOM 560M/1.1B/1.7B). For each model it loads prompts and
reference answers from the sampled dataset, builds the transition graph per prompt (top-90
candidates), computes the same set of graph metrics plus a word-overlap accuracy score against the
reference answers, averages everything across prompts, and saves both a per-model metrics JSON and
a per-model generated-output JSON. It finishes by producing one bar chart per metric comparing all
13 models, plus a combined `model_metrics.json`.

### `results/`

The metrics JSON files and PNG plots that `training_dynamics.py` and `multi_model_eval.py`
actually produced for the paper, organized per dataset. This is the output you should be able to
reproduce by re-running the two scripts above against the same `sampled_datasets/` files.


## `graph_metrics_eval.py` — General-Purpose CLI Tool

This script supports two evaluation modes and two dataset sources, controlled entirely through
command-line arguments.

## Requirements

```bash
pip install torch transformers datasets networkx matplotlib numpy
```

A CUDA-capable GPU is recommended but not required; the script falls back to CPU automatically.

## Modes

### `checkpoints` mode

Tracks how the metrics evolve across training checkpoints of a single Pythia model
(`EleutherAI/pythia-<size>-deduped`). Requires `--model_size`.

### `models` mode

Compares the metrics across an arbitrary list of pretrained causal LM checkpoints
(any model on the Hugging Face Hub). Requires `--models`.

## Datasets

### Predefined datasets

Pass one of: `logiqa`, `piqa`, `arc`. The script downloads the corresponding Hugging Face
dataset, randomly samples `--num_samples` examples from its validation/test split, and formats
each one into a prompt automatically.

### Custom dataset

Pass `--dataset custom --custom_dataset path/to/file.json`. The file must follow this structure:

```json
{
  "samples": [
    { "prompt": "Your prompt text here", "references": ["expected answer text"] }
  ]
}
```

`references` is optional and only used for record-keeping in the saved output files; it is not
used to compute any of the graph metrics. See `custom_dataset_example.json` in this folder for a
working example covering open-ended, multiple-choice, and two-option prompt styles.

## Parameters

| Argument | Required | Applies to | Description |
|---|---|---|---|
| `--mode` | yes | both | `checkpoints` or `models` |
| `--dataset` | yes | both | `logiqa`, `piqa`, `arc`, or `custom` |
| `--custom_dataset` | only if `--dataset custom` | both | Path to your custom dataset JSON file |
| `--num_samples` | no | predefined datasets only | How many examples to sample (default `5`) |
| `--seed` | no | predefined datasets only | Random seed for sampling (default `42`) |
| `--models` | only if `--mode models` | models | One or more Hugging Face model IDs, space separated |
| `--model_base` | no | checkpoints | Base model namespace (default `EleutherAI/pythia`) |
| `--model_size` | only if `--mode checkpoints` | checkpoints | e.g. `70m`, `160m`, `410m`, `1b`, `1.4b`, `2.8b`, `6.9b`, `12b` |
| `--steps` | no | checkpoints | Checkpoint revisions to evaluate, space separated (defaults to 10 evenly spread training steps) |
| `--token_limit` | no | both | Max new tokens generated per prompt (default `200`) |
| `--top_k` | no | both | Number of candidate tokens recorded per generation step (default `60`) |
| `--metrics` | no | both | Subset of `spectral_entropy graph_energy density` to compute and save (default: all three) |
| `--output_dir` | no | both | Where metrics, generated outputs, and plots are written (default `results`) |
| `--cache_dir` | no | both | Where downloaded model weights are cached (default `.`) |

## Example usage

Track metrics across Pythia-410m training checkpoints on the ARC dataset:

```bash
python graph_metrics_eval.py \
  --mode checkpoints \
  --dataset arc \
  --model_size 410m \
  --num_samples 5 \
  --output_dir results_pythia410m_arc
```

Compare metrics across several different model families on a custom dataset:

```bash
python graph_metrics_eval.py \
  --mode models \
  --dataset custom \
  --custom_dataset custom_dataset_example.json \
  --models facebook/opt-350m EleutherAI/pythia-410m bigscience/bloom-560m \
  --output_dir results_model_comparison
```

Only compute and save a subset of metrics:

```bash
python graph_metrics_eval.py \
  --mode models \
  --dataset piqa \
  --models EleutherAI/pythia-160m \
  --metrics spectral_entropy density \
  --output_dir results_piqa_subset
```

## Output

Each run writes the following into `--output_dir`:

- `metrics_<step_or_model>.json` — averaged metric values for that checkpoint/model
- `outputs_<step_or_model>.json` — the generated text and reference answers for every prompt
- `metrics_all_steps.json` (checkpoints mode) or `metrics_all_models.json` (models mode) — combined results
- PNG plots: one line chart per metric across steps (checkpoints mode), or one bar chart per metric across models (models mode)
## Questions & Support

If you run into any issues running the code, getting the paper results to reproduce, or have any
other questions about this repository, feel free to email me: rahnamounrashin@gmail.com
## Notes

- `checkpoints` mode assumes the model family exposes its training checkpoints as Hugging Face
  `revision` tags (this is the convention used by the EleutherAI Pythia suite).
- `models` mode works with any causal LM on the Hub, since each model is loaded independently
  with no `revision` argument.
- Predefined dataset downloads are cached under `--cache_dir` for subsequent runs.
