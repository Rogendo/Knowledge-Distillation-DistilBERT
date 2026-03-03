# Attention Fusion Multi-Task NLP Model

A single DistilBERT backbone that simultaneously handles three tasks with
**~3× fewer parameters** and **one forward pass** compared to running three
separate models:

| Task | Type | Output |
|------|------|--------|
| Named Entity Recognition (NER) | Token-level | 10 entity labels |
| Case Classification (CLS) | Sentence-level | 4 heads: main category, sub-category, intervention, priority |
| Quality Assurance Scoring (QA) | Sentence-level | 6 binary heads (17 sub-metrics total) |

---

## Table of Contents

1. [Architecture](#architecture)
2. [Repository Layout](#repository-layout)
3. [Prerequisites](#prerequisites)
4. [Training](#training)
5. [Configuration Reference](#configuration-reference)
6. [MLflow Experiment Tracking](#mlflow-experiment-tracking)
7. [Checkpoint Output](#checkpoint-output)
8. [Inference](#inference)
9. [Parameter Count Comparison](#parameter-count-comparison)
10. [Task-Alternation Training Strategy](#task-alternation-training-strategy)
11. [Troubleshooting](#troubleshooting)

---

## Architecture

```
Input Text
    │
    ▼
[Shared DistilBERT-base-cased backbone]
    │
Hidden States  [batch, seq_len, 768]
    │
    ├─────────────────┬──────────────────┬──────────────────┐
    │                 │                  │                  │
  task='ner'      task='cls'         task='qa'             │
    │                 │                  │                  │
┌───────────┐  ┌──────────────┐  ┌──────────────┐         │
│ NER Head  │  │  CLS Head    │  │   QA Head    │         │
│           │  │              │  │              │         │
│ Linear    │  │ TaskAttn     │  │ TaskAttn     │         │
│ (768→768) │  │ Pooling      │  │ Pooling      │         │
│ + GELU    │  │ (learned     │  │ (learned     │         │
│ [tokens]  │  │  weighted    │  │  weighted    │         │
│     ↓     │  │  sum over    │  │  sum over    │         │
│ Linear    │  │  seq_len)    │  │  seq_len)    │         │
│ (768→10)  │  │     ↓        │  │     ↓        │         │
│ per token │  │  Dropout     │  │  Dropout     │         │
│           │  │     ↓        │  │     ↓        │         │
│ CE Loss   │  │ main   (→8)  │  │ opening (→1) │         │
│ ignore    │  │ sub    (→77) │  │ listen  (→5) │         │
│ =-100     │  │ interv (→16) │  │ proact  (→3) │         │
└───────────┘  │ prior  (→3)  │  │ resolv  (→5) │         │
               │              │  │ hold    (→2) │         │
               │ CE Loss × 4  │  │ closing (→1) │         │
               └──────────────┘  │ BCE × 6      │         │
                                 └──────────────┘         │
```

### TaskAttentionPooling

Each sentence-level task (CLS, QA) uses its own learned attention pooling layer
instead of the fixed `[CLS]` token. This lets each task learn *which tokens in the
transcript are most informative for its objective*.

```python
# Simplified
scores  = Linear(768 → 1)(hidden_states)          # [batch, seq_len, 1]
scores  = masked_fill(padding positions, -inf)     # ignore padding
weights = softmax(scores, dim=seq_len)             # [batch, seq_len, 1]
pooled  = (weights * hidden_states).sum(dim=1)     # [batch, 768]
```

The NER task operates at token level and does not use attention pooling.

---

## Repository Layout

```
attention_fusion/
├── config.yaml          ← all training hyperparameters
├── model.py             ← AttentionFusionModel, TaskAttentionPooling
├── datasets.py          ← NERDataset, ClassificationDataset, QADataset,
│                           build_label_maps, MultiTaskDataModule,
│                           filter_by_token_length, collate functions
├── trainer.py           ← MultiTaskTrainer (task-alternation loop,
│                           early stopping, MLflow logging)
├── inference.py         ← AttentionFusionInference
└── attention_fusion_model/   ← written after training
    ├── model.pt
    ├── label_maps.json
    ├── metrics.json
    └── tokenizer files (vocab.txt, tokenizer.json, ...)

../data/
├── ner_synthetic_dataset_v1.jsonl
├── cases_generated_data_v0005.json
└── synthetic_qa_metrics_data_v01x.json
```

---

## Prerequisites

```bash
# Activate your virtual environment
source /path/to/your/venv/bin/activate

# Verify required packages are present
python -c "import torch, transformers, mlflow, sklearn, yaml; print('All OK')"
```

Install any missing packages:

```bash
pip install torch transformers scikit-learn mlflow pyyaml tqdm pandas
```

---

## Training

### 1. Navigate to this directory

```bash
cd /home/rogendo/Desktop/Knowledge-Distillation-DistilBERT/attention_fusion
```

### 2. Run training

```bash
python trainer.py --config config.yaml
```

### 3. What happens during training

1. **Label maps** are built from the three raw data files and saved to
   `./attention_fusion_model/label_maps.json`. These are required at inference time.

2. **Tokenizer** (`distilbert-base-cased`) is downloaded on first run and saved
   to `./attention_fusion_model/` so inference does not need internet access.

3. **Data preparation** per task:
   - **NER** — records whose tokenized text exceeds `max_length` are **filtered out**
     (not truncated). Truncation would silently drop entity labels at the tail of the
     sequence. ~99.9% of records are kept.
   - **CLS / QA** — long records are **truncated** at `max_length`. Labels are
     sentence-level and are unaffected by truncation.

4. **Task-alternation training** (see [Task-Alternation Training Strategy](#task-alternation-training-strategy)):
   All batches from all three loaders are shuffled together each epoch, giving the
   shared backbone gradients from every task continuously.

5. **Validation** after each epoch reports three metrics:
   - `ner_f1` — macro F1 over non-padding tokens
   - `cls_avg_acc` — average accuracy across 4 classification heads
   - `qa_avg_f1` — average micro-F1 across 6 QA binary heads
   - `avg` — arithmetic mean of the three (used for early stopping and checkpointing)

6. **Best checkpoint** (by `avg`) is saved to `./attention_fusion_model/`. All
   metrics are also logged to MLflow.

7. **Early stopping** halts training if no improvement for
   `early_stopping_patience` consecutive epochs.

### 4. Expected console output

```
Label maps saved to ./attention_fusion_model/label_maps.json
Label counts — main: 8, sub: 77, interv: 16, priority: 3

NER  token filter : kept 1998/2000 (99.9%) — 2 over-length removed
CLS  truncation   : 9996 records kept, long narratives truncated at 512
QA   truncation   : 4996 records kept, long transcripts truncated at 512

Final dataset sizes:
  NER: 1798 train / 200 val
  CLS: 8996 train / 1000 val
  QA:  4496 train / 500 val

Total parameters: 65,883,779
  [MLflow] Run started: multitask_train (id=7848b563...)
  [MLflow] Tracking URI: file:./mlruns
Training on cuda
Epochs: 10, patience: 3

Epoch 1/10
  Training: 100%|████| 955/955 [03:21<00:00]
  train_loss=2.3141  ner=1.2134  cls=0.9823  qa=0.1184
  Val — NER F1: 0.412  CLS acc: 0.531  QA F1: 0.483  avg: 0.475
  [New best — checkpoint saved]
  [MLflow] epoch metrics logged.

Epoch 2/10
  ...
```

---

## Configuration Reference

All settings live in `config.yaml`:

```yaml
backbone:
  model_name: "distilbert-base-cased"
  # Use distilbert-base-cased (not uncased) — NER is case-sensitive.
  # Capitalisation signals (e.g. "John" vs "john") matter for entity detection.
  dropout: 0.1

data:
  ner_path:            "../data/ner_synthetic_dataset_v1.jsonl"
  classification_path: "../data/cases_generated_data_v0005.json"
  qa_path:             "../data/synthetic_qa_metrics_data_v01x.json"
  test_size:  0.1       # fraction held out as validation set
  random_seed: 42

tokenizer:
  max_length: 512       # DistilBERT hard limit — do not increase beyond 512
  padding:    "max_length"
  truncation: true

training:
  num_epochs:              10
  batch_size:              16   # reduce to 8 or 4 if CUDA OOM
  learning_rate:           2.0e-5
  weight_decay:            0.01
  warmup_ratio:            0.1  # 10% of total steps used for LR warm-up
  max_grad_norm:           1.0  # gradient clipping
  early_stopping_patience: 3

output:
  model_dir:       "./attention_fusion_model"
  label_maps_file: "label_maps.json"
  metrics_file:    "metrics.json"

mlflow:
  enabled:          true
  tracking_uri:     "file:./mlruns"
  # Change to "http://<server>:5000" to use a remote tracking server.
  # If the remote server is unreachable, training falls back to file:./mlruns
  # automatically — training is never blocked by MLflow connectivity.
  experiment_name:  "attention_fusion_multitask"
  run_name:         "multitask_train"
```

---

## MLflow Experiment Tracking

Every training run is automatically logged to the `./mlruns/` directory (or to
the remote tracking server specified in `config.yaml`). This section explains
everything that is logged, how to explore it in the UI, how to query it
programmatically, and how to restore a model from a logged artifact.

---

### What is logged per run

#### Parameters (logged once at run start)

Hyperparameters and data-preparation statistics that define exactly what was
trained:

| Parameter key | Example value | Description |
|---------------|:---:|---------|
| `backbone` | `distilbert-base-cased` | HuggingFace model ID |
| `learning_rate` | `2e-05` | Peak LR after warm-up |
| `batch_size` | `16` | Samples per gradient step |
| `num_epochs` | `10` | Maximum training epochs |
| `weight_decay` | `0.01` | AdamW L2 regularisation |
| `warmup_ratio` | `0.1` | Fraction of total steps used for warm-up |
| `max_grad_norm` | `1.0` | Gradient clipping threshold |
| `early_stopping_patience` | `3` | Epochs without improvement before stopping |
| `max_length` | `512` | Tokenizer max sequence length |
| `filter_ner_before` | `2000` | NER records before token-length filtering |
| `filter_ner_after` | `1998` | NER records kept after filtering |
| `filter_ner_pct_kept` | `99.9` | Percentage retained |
| `filter_cls_before` | `9996` | CLS records before truncation |
| `filter_qa_before` | `4996` | QA records before truncation |

#### Tags (logged once at run start)

| Tag key | Example value | Description |
|---------|:---:|---------|
| `device` | `cuda` | Hardware used for training |
| `total_params` | `65,883,779` | Total trainable parameter count |
| `tasks` | `ner,cls,qa` | Tasks trained in this run |
| `strategy` | `task_alternation` | Batch sampling strategy |
| `stopped_early` | `true` / `false` | Whether early stopping triggered |

#### Metrics (logged every epoch)

Training metrics — mean across all batches in the epoch:

| Metric key | Description |
|------------|-------------|
| `train/loss` | Combined loss across all tasks |
| `train/loss_ner` | NER task loss (CrossEntropy, token-level) |
| `train/loss_cls` | CLS task loss (sum of 4 CrossEntropy heads) |
| `train/loss_qa` | QA task loss (sum of 6 BCE heads) |

Validation metrics — computed on held-out val sets:

| Metric key | Description |
|------------|-------------|
| `val/ner_f1` | Macro F1 over non-padding NER tokens |
| `val/cls_avg_acc` | Average accuracy across 4 classification heads |
| `val/qa_avg_f1` | Average micro-F1 across 6 QA binary heads (threshold=0.5) |
| `val/avg` | Arithmetic mean of the 3 val metrics — used for checkpointing |
| `best/avg_val` | Running best `val/avg` (updated only when a new best is reached) |

#### Artifacts (logged only when a new best checkpoint is saved)

| Artifact | Description |
|----------|-------------|
| `model.pt` | PyTorch state dict of the best checkpoint |
| `label_maps.json` | All label→ID mappings for NER, CLS, and QA |
| `metrics.json` | Val metrics at the time of the best checkpoint |

---

### Launching the MLflow UI

#### Local file store (default)

```bash
cd /home/rogendo/Desktop/Knowledge-Distillation-DistilBERT/attention_fusion

mlflow ui --backend-store-uri file:./mlruns
# UI available at http://127.0.0.1:5000
```

Or, if you want to browse from another machine on your network:

```bash
mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5000
```

#### Remote tracking server

If `tracking_uri` in `config.yaml` points to a running MLflow server, the UI
is already available at that address — no extra command needed.

---

### Starting the MLflow server

#### Option A — lightweight built-in UI (read-only, no server process)

Use `mlflow ui` for a quick local browser view. It reads directly from the file
store and requires no server process to be running beforehand.

```bash
# IMPORTANT: run this from the attention_fusion/ directory,
# NOT from inside the mlruns/ subdirectory.
cd /home/rogendo/Desktop/Knowledge-Distillation-DistilBERT/attention_fusion

mlflow ui --backend-store-uri file:./mlruns --port 5000
# Open http://127.0.0.1:5000 in your browser
```

#### Option B — full MLflow server (supports concurrent training runs + REST API)

Use `mlflow server` when you want the tracking server accessible from other
machines on your network, or when multiple training jobs will log to it at the
same time.

```bash
# Run from the attention_fusion/ directory
cd /home/rogendo/Desktop/Knowledge-Distillation-DistilBERT/attention_fusion

mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri file:///home/rogendo/Desktop/Knowledge-Distillation-DistilBERT/attention_fusion/mlruns
```

- `--host 0.0.0.0` makes the UI reachable from any machine on the local network
  (e.g. `http://192.168.8.18:5000` from another device).
- `--host 127.0.0.1` (or omit `--host`) restricts access to localhost only.
- The absolute path in `--backend-store-uri` avoids ambiguity regardless of
  which directory you launch from.

Then update `config.yaml` so training runs log to this server instead of the
local file store:

```yaml
mlflow:
  tracking_uri: "http://0.0.0.0:5000"   # or use your machine's LAN IP
```

#### Common pitfall — running from inside `mlruns/`

If you `cd mlruns/` and then run:

```bash
# WRONG — do not do this
mlflow server --backend-store-uri file:///…/attention_fusion/mlruns
```

MLflow will try to read experiments from `mlruns/mlruns/` (the path appended to
your current working directory), which does not exist. You will see:

```
WARNING: Malformed experiment 'mlruns'. Detailed error
Yaml file '…/mlruns/mlruns/meta.yaml' does not exist.
```

**Fix**: always run the server from the `attention_fusion/` directory, and use
the absolute path to `mlruns/` in `--backend-store-uri`.

#### Running as a background process

```bash
cd /home/rogendo/Desktop/Knowledge-Distillation-DistilBERT/attention_fusion

nohup mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri file:///home/rogendo/Desktop/Knowledge-Distillation-DistilBERT/attention_fusion/mlruns \
  > mlflow_server.log 2>&1 &

echo "MLflow server PID: $!"
# Logs written to attention_fusion/mlflow_server.log
# Stop with: kill <PID>
```

---

### Navigating the MLflow UI

Once the UI is open at `http://127.0.0.1:5000` (or your server's LAN IP):

#### 1. Find your experiment

The left sidebar lists all experiments. Click **attention_fusion_multitask** to
see all runs.

```
Experiments
└── attention_fusion_multitask          ← click here
    ├── multitask_train  (run 1)
    ├── multitask_train  (run 2)
    └── ...
```

#### 2. Runs table

The runs table shows every training run with its final metric values. Columns can
be sorted and filtered. Key columns to look at:

- `val/avg` — composite validation score (higher is better)
- `val/ner_f1`, `val/cls_avg_acc`, `val/qa_avg_f1` — per-task breakdown
- `train/loss` — final epoch training loss
- `Status` — `FINISHED`, `RUNNING`, or `FAILED`

#### 3. Comparing runs

1. Select two or more runs using the checkboxes on the left.
2. Click **Compare** at the top of the table.
3. The comparison view shows:
   - A **parameters diff** table highlighting where runs differ
   - **Metric curves** — side-by-side time-series charts for every logged metric
   - A **parallel coordinates plot** — useful for spotting correlations between
     hyperparameters and final metrics

#### 4. Inspecting a single run

Click any run name to open its detail page:

```
Run detail page
├── Overview       ← run ID, start time, duration, git commit
├── Parameters     ← all logged params in a searchable table
├── Metrics        ← interactive time-series chart per metric
│   ├── train/loss          (click to zoom, hover for exact values)
│   ├── val/ner_f1
│   ├── val/cls_avg_acc
│   └── val/avg
├── Tags           ← device, strategy, stopped_early, etc.
└── Artifacts      ← downloadable files: model.pt, label_maps.json, metrics.json
```

**To view a metric chart**: click the metric name under the **Metrics** tab. Each
data point is one epoch. Hover to see the exact value and step number.

**To download an artifact**: click the artifact name under the **Artifacts** tab,
then click the download icon.

---

### Querying runs programmatically

Use the MLflow Python client to search and retrieve runs without opening the UI:

```python
import mlflow
from mlflow.tracking import MlflowClient

# Point to the same store used during training
mlflow.set_tracking_uri("file:./mlruns")
client = MlflowClient()

# Get the experiment by name
experiment = client.get_experiment_by_name("attention_fusion_multitask")
exp_id = experiment.experiment_id

# List all finished runs, sorted by val/avg descending
runs = client.search_runs(
    experiment_ids=[exp_id],
    filter_string="status = 'FINISHED'",
    order_by=["metrics.`val/avg` DESC"],
)

for run in runs:
    m = run.data.metrics
    p = run.data.params
    print(
        f"run_id={run.info.run_id[:8]}  "
        f"val/avg={m.get('val/avg', 0):.4f}  "
        f"ner_f1={m.get('val/ner_f1', 0):.4f}  "
        f"cls_acc={m.get('val/cls_avg_acc', 0):.4f}  "
        f"qa_f1={m.get('val/qa_avg_f1', 0):.4f}  "
        f"lr={p.get('learning_rate')}  "
        f"bs={p.get('batch_size')}"
    )
```

**Get the metric history for a specific run** (all epochs):

```python
run_id = runs[0].info.run_id   # best run

history = client.get_metric_history(run_id, "val/avg")
for point in history:
    print(f"  epoch {point.step}: val/avg = {point.value:.4f}")
```

**Download the best checkpoint's artifacts**:

```python
import os

best_run_id = runs[0].info.run_id
artifact_dir = f"./restored_from_mlflow/{best_run_id}"
os.makedirs(artifact_dir, exist_ok=True)

# Download model.pt
client.download_artifacts(best_run_id, "model.pt", artifact_dir)

# Download label_maps.json
client.download_artifacts(best_run_id, "label_maps.json", artifact_dir)

print(f"Artifacts saved to {artifact_dir}/")
```

---

### Restoring a model from MLflow artifacts

Once you have downloaded `model.pt` and `label_maps.json` (either via the UI or
the client above), restore the full model for inference:

```python
import json
import torch
from model import AttentionFusionModel

# Load label maps to get head sizes
with open("label_maps.json") as f:
    label_maps = json.load(f)

num_ner    = len(label_maps["ner"]["label_to_id"])
num_main   = len(label_maps["cls"]["main_cat2id"])
num_sub    = len(label_maps["cls"]["sub_cat2id"])
num_interv = len(label_maps["cls"]["interv2id"])
num_prio   = len(label_maps["cls"]["priority2id"])
qa_heads   = label_maps["qa"]["head_config"]

# Rebuild the model architecture
model = AttentionFusionModel(
    model_name="distilbert-base-cased",
    num_ner_labels=num_ner,
    num_main_categories=num_main,
    num_sub_categories=num_sub,
    num_interventions=num_interv,
    num_priorities=num_prio,
    qa_heads_config=qa_heads,
)

# Load weights
state_dict = torch.load("model.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
print("Model restored successfully.")
```

Or use the high-level `AttentionFusionInference` wrapper which handles all of
the above automatically (see [Inference](#inference)).

---

### Using a remote MLflow tracking server

Set `tracking_uri` in `config.yaml` to the server address:

```yaml
mlflow:
  enabled: true
  tracking_uri: "http://192.168.8.18:5000"
  experiment_name: "attention_fusion_multitask"
  run_name: "multitask_train"
```

The training script implements automatic fallback — if the remote server is
unreachable at startup, it switches to `file:./mlruns` so training is never
blocked:

```python
# trainer.py fallback logic
try:
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    client = MlflowClient()
    client.search_experiments()          # connectivity check
except Exception:
    fallback = "file:./mlruns"
    mlflow.set_tracking_uri(fallback)
    print(f"[MLflow] Remote unreachable — falling back to {fallback}")
```

---

### Disabling MLflow

Set `enabled: false` in `config.yaml` to skip all MLflow calls. Training and
checkpointing continue normally; nothing is logged.

```yaml
mlflow:
  enabled: false
```

---

## Checkpoint Output

After training, `./attention_fusion_model/` contains:

```
attention_fusion_model/
├── model.pt              ← best model weights (PyTorch state dict)
├── label_maps.json       ← all label→id mappings for all three tasks
├── metrics.json          ← val metrics recorded at the best checkpoint
├── tokenizer_config.json
├── vocab.txt
├── tokenizer.json
└── special_tokens_map.json
```

`label_maps.json` structure:

```json
{
  "ner": {
    "label_to_id": {"O": 0, "NAME": 1, "LOCATION": 2, ...},
    "id_to_label": {"0": "O", "1": "NAME", "2": "LOCATION", ...}
  },
  "cls": {
    "main_cat2id":  {"Advice and Counselling": 0, "GBV": 1, ...},
    "sub_cat2id":   {"Physical Abuse": 0, "Drug/Alcohol Abuse": 1, ...},
    "interv2id":    {"Counselling": 0, "Referral": 1, ...},
    "priority2id":  {"1": 0, "2": 1, "3": 2},
    "id_to_main":   {"0": "Advice and Counselling", ...},
    "id_to_sub":    {"0": "Physical Abuse", ...},
    "id_to_interv": {"0": "Counselling", ...},
    "id_to_priority":{"0": "1", ...}
  },
  "qa": {
    "head_config": {
      "opening": 1, "listening": 5, "proactiveness": 3,
      "resolution": 5, "hold": 2, "closing": 1
    }
  }
}
```

---

## Inference

### Command-line smoke test

```bash
python inference.py --model_dir ./attention_fusion_model
```

### Programmatic use

```python
from inference import AttentionFusionInference

# Load from saved checkpoint
inf = AttentionFusionInference(
    model_dir="./attention_fusion_model",
    device="cuda",        # or "cpu"
    qa_threshold=0.5,     # sigmoid threshold for QA binary predictions
)

texts = [
    "I'm calling from Nairobi. My daughter Sarah, aged 12, was assaulted by her teacher.",
]

# Named Entity Recognition
ner_results = inf.predict_ner(texts)
# → [[("I", "O"), ("'m", "O"), ..., ("Sarah", "NAME"), (",", "O"),
#     ("aged", "O"), ("12", "AGE"), ..., ("Nairobi", "LOCATION"), ...]]

# Case Classification
cls_results = inf.predict_classification(texts)
# → [{"main_category": "GBV",
#     "sub_category": "Physical Abuse",
#     "intervention": "Referral to Social Services",
#     "priority": "1"}]

# Quality Assurance Scoring
qa_results = inf.predict_qa(texts)
# → [{"opening":      [1],
#     "listening":    [1, 0, 1, 1, 0],
#     "proactiveness":[0, 1, 0],
#     "resolution":   [1, 1, 0, 0, 1],
#     "hold":         [0, 0],
#     "closing":      [1]}]
```

The QA output is a dict of binary lists — `1` means the sub-metric criterion was
met, `0` means it was not.

**QA head sub-metrics** (in order):

| Head | Sub-metrics |
|------|-------------|
| `opening` (1) | Use of call opening phrase |
| `listening` (5) | Caller not interrupted · Empathises · Paraphrases issue · Uses please/thank you · Doesn't hesitate |
| `proactiveness` (3) | Solves extra issues · Confirms satisfaction · Follows up on updates |
| `resolution` (5) | Accurate info · Correct language · Consults if unsure · Follows steps · Explains solution |
| `hold` (2) | Explains before hold · Thanks for holding |
| `closing` (1) | Proper call closing phrase |

---

## Parameter Count Comparison

| Setup | Parameters | Forward passes per text |
|-------|:---:|:---:|
| 3 separate DistilBERT models | ~200M | 3 |
| **Attention Fusion (this model)** | **~66M** | **1** |

The three task-specific heads (NER, CLS, QA) add only ~3M parameters on top of
the 63M backbone.

---

## Task-Alternation Training Strategy

### The problem with sequential task training

A naive multi-task approach trains one task per epoch in rotation:
`NER epoch → CLS epoch → QA epoch → repeat`. This causes **catastrophic
forgetting**: training exclusively on CLS data overwrites NER-specific patterns
in the shared backbone. By the time QA training begins, the backbone has already
drifted from what NER needs.

### The solution: interleaved batch shuffling

At the start of every epoch, all batches from all three DataLoaders are tagged
with their task name and placed into a single list. That list is then randomly
shuffled before iteration begins:

```
Epoch N
│
├── Collect all batches
│   ├── NER   loader  → ~112 batches  tagged ('ner',  batch)
│   ├── CLS   loader  → ~562 batches  tagged ('cls',  batch)
│   └── QA    loader  → ~281 batches  tagged ('qa',   batch)
│                                     ─────────────────────
│                               total ~955 batches
│
├── random.shuffle(all_batches)
│   → ('cls', b1), ('ner', b2), ('qa', b3), ('cls', b4), ('ner', b5), ...
│
└── Iterate:
    for task, batch in all_batches:
        loss = model(batch, task=task)
        loss.backward()
        optimizer.step()
        scheduler.step()
```

### Why this works

- **No task is starved**: The backbone never goes more than a few steps without
  a gradient from each task.
- **Prevents catastrophic forgetting**: Because a QA batch may immediately follow
  a NER batch, the backbone cannot drift far from any task's requirements.
- **Better backbone regularisation**: The shared weights must simultaneously
  satisfy constraints from three different tasks, which acts as a natural
  regulariser and reduces overfitting to any single task.
- **Respects dataset proportions**: CLS (~562 batches) naturally contributes more
  than NER (~112 batches), proportional to dataset size, without any manual
  loss weighting.

### Learning rate schedule

A single `get_linear_schedule_with_warmup` scheduler covers the entire run:

```
total_steps  = (NER + CLS + QA batches per epoch) × num_epochs
             = ~955 × 10 = ~9,550 steps
warmup_steps = 10% of total_steps = ~955 steps

LR:  0 ──warm-up──► 2e-5 ──linear decay──► 0
     |←── 955 ────►|←──────── 8,595 ────────►|
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Reduce `batch_size` to `8` or `4` in `config.yaml` |
| `MLflow connection refused` | Set `tracking_uri: "file:./mlruns"` in `config.yaml`, or check that the remote MLflow server is running |
| `ModuleNotFoundError` | Activate your virtual environment and install dependencies |
| NER F1 stays near 0 for first 1–2 epochs | Normal — NER is the hardest task and typically lags CLS/QA early |
| Early stopping triggers too soon | Increase `early_stopping_patience` from `3` to `5` in `config.yaml` |
| `mlflow ui` shows no runs | Confirm `--backend-store-uri` matches the `tracking_uri` used during training |
| `Malformed experiment 'mlruns'` / `meta.yaml does not exist` | You ran `mlflow server` from inside `mlruns/` — the path doubled up to `mlruns/mlruns/`. Run the server from `attention_fusion/` using the absolute path to `mlruns/` |
| Metrics tab empty in UI | The run may still be `RUNNING` — refresh, or check if training crashed mid-epoch |
| `JSONDecodeError` on startup | A data file was partially written; re-run once the file is fully written |
| `ValueError: too many dimensions 'str'` in QA loader | A label list contains a non-numeric sentinel (e.g. `"No hold was required"`); the dataset class coerces these to `0.0` — ensure you are running the latest `datasets.py` |
