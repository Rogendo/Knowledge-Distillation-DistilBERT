# Attention Fusion Multi-Task NLP Model

A single DistilBERT backbone that simultaneously handles three tasks with
**~3× fewer parameters** and **one forward pass** compared to running three
separate models:

| Task | Type | Output |
|------|------|--------|
| Named Entity Recognition (NER) | Token-level | 10 entity labels |
| Case Classification (CLS) | Sentence-level | 4 heads: main category, sub-category, intervention, priority |
| Quality Assurance scoring (QA) | Sentence-level | 6 binary heads (17 sub-metrics total) |

---

## Architecture

```
Input Text
    ↓
[Shared DistilBERT-base-cased backbone]
    ↓
Hidden States  [batch, seq_len, 768]
    ↓
┌─────────────────┬──────────────────┬──────────────────┐
│  NER            │  CLS             │  QA              │
│                 │                  │                  │
│  Linear(768→    │  TaskAttention   │  TaskAttention   │
│  768) + GELU    │  Pooling         │  Pooling         │
│  [all tokens]   │  (learned attn   │  (learned attn   │
│       ↓         │   weighted sum)  │   weighted sum)  │
│  Linear(768,10) │       ↓          │       ↓          │
│  per token      │  main_cat  ( 8)  │  opening    (1)  │
│                 │  sub_cat   (77)  │  listening  (5)  │
│  CrossEntropy   │  interven  (16)  │  proactive  (3)  │
│                 │  priority  ( 3)  │  resolution (5)  │
│                 │                  │  hold       (2)  │
│                 │  CrossEntropy×4  │  closing    (1)  │
│                 │                  │  BCE×6           │
└─────────────────┴──────────────────┴──────────────────┘
```

**TaskAttentionPooling** — each sentence-level task has its own learned
attention layer that selectively weights tokens, so each task learns *which
tokens matter* for its predictions (rather than always using the fixed [CLS]
token).

---

## Repository layout

```
scripts/
├── attention_fusion/
│   ├── config.yaml        ← all training hyperparameters
│   ├── model.py           ← AttentionFusionModel + TaskAttentionPooling
│   ├── datasets.py        ← NERDataset, ClassificationDataset, QADataset,
│   │                         build_label_maps, MultiTaskDataModule,
│   │                         filter_by_token_length
│   ├── trainer.py         ← MultiTaskTrainer (task-alternation loop,
│   │                         early stopping, MLflow logging)
│   └── inference.py       ← AttentionFusionInference
└── data/
    ├── ner_synthetic_dataset_v1.jsonl
    ├── cases_generated_data_v0005.json
    └── synthetic_qa_metrics_data_v01x.json
```

---

## Prerequisites

The project uses the virtual environment at `/home/naynek/Work/ai/venv`.

```bash
# Activate the environment
source /home/naynek/Work/ai/venv/bin/activate

# Verify required packages
python -c "import torch, transformers, mlflow, sklearn; print('OK')"
```

If `mlflow` is missing:
```bash
pip install mlflow
```

---

## Training

### 1. Navigate to the module directory

```bash
cd /home/naynek/Desktop/Jenga-AI/scripts/attention_fusion
```

### 2. Run training

```bash
python trainer.py --config config.yaml
```

Or use the virtual environment python directly without activating:

```bash
/home/naynek/Work/ai/venv/bin/python trainer.py --config config.yaml
```

### 3. What happens during training

1. Label maps are built from the raw data files and saved to
   `./attention_fusion_model/label_maps.json`.
2. The tokenizer (`distilbert-base-cased`) is downloaded on first run and
   saved to `./attention_fusion_model/` for use at inference time.
3. Each dataset is prepared:
   - **NER** — records longer than `max_length` tokens are **filtered out**
     (truncation would silently drop entity labels at the end of the text).
   - **CLS / QA** — long records are **truncated** at `max_length` (sentence-
     level labels are unaffected by truncation).
4. Training runs for up to `num_epochs` epochs using **task alternation**:
   all batches from all three loaders are shuffled together each epoch, so
   the shared backbone receives gradients from every task continuously.
5. Validation metrics are printed each epoch:
   - `ner_f1` — macro F1 over non-padding NER tokens
   - `cls_avg_acc` — average accuracy across the 4 classification heads
   - `qa_avg_f1` — average micro-F1 across the 6 QA binary heads
6. The best checkpoint (by average of the three val metrics) is saved to
   `./attention_fusion_model/`.
7. Training stops early if no improvement for `early_stopping_patience`
   consecutive epochs.

### 4. Expected console output

```
NER  token filter : kept 1998/2000 (99.9%) — 2 over-length removed
CLS  truncation   : 9996 records kept, long narratives truncated at 512
QA   truncation   : 4996 records kept, long transcripts truncated at 512

Final dataset sizes:
  NER: 1798 train / 200 val
  CLS: 8996 train / 1000 val
  QA:  4496 train / 500 val

Total parameters: ~67,000,000
Training on cpu   (or cuda if a GPU is available)

Epoch 1/10
  Train loss: 2.3412  ner=1.2134  cls=0.9823  qa=0.1455
  Val: ner_f1=0.4821  cls_avg_acc=0.3102  qa_avg_f1=0.6214  avg=0.4712
  New best avg val metric: 0.4712
  Checkpoint saved to ./attention_fusion_model
  [MLflow] Run finalised.
```

---

## Configuration

All settings live in `config.yaml`. Key options:

```yaml
backbone:
  model_name: "distilbert-base-cased"  # HuggingFace model ID
  dropout: 0.1

tokenizer:
  max_length: 512    # DistilBERT hard limit — do not increase beyond 512

training:
  num_epochs: 10
  batch_size: 16     # reduce to 8 if running out of memory on CPU
  learning_rate: 2.0e-5
  early_stopping_patience: 3

mlflow:
  enabled: true
  tracking_uri: "file:./mlruns"   # local file store, no server needed
                                  # change to "http://localhost:5000" for
                                  # a remote MLflow tracking server
  experiment_name: "attention_fusion_multitask"
  run_name: "multitask_train"
```

---

## MLflow experiment tracking

Every training run is automatically logged to `./mlruns/`.

### Viewing the MLflow UI

```bash
cd /home/naynek/Desktop/Jenga-AI/scripts/attention_fusion
mlflow ui
# Opens at http://127.0.0.1:5000
```

### What is logged per run

| Category | Logged items |
|----------|-------------|
| **Params** | All hyperparameters, token filter stats (records kept/removed per task) |
| **Tags** | Device, total param count, training strategy, stopped_early flag |
| **Metrics (per epoch)** | `train/loss`, `train/loss_ner`, `train/loss_cls`, `train/loss_qa` |
| | `val/ner_f1`, `val/cls_avg_acc`, `val/qa_avg_f1`, `val/avg` |
| | `best/avg_val` (updated only when a new best is reached) |
| **Artifacts** | `model.pt`, `label_maps.json`, `metrics.json` (best checkpoint only) |

### Connecting to a remote MLflow server

```yaml
# config.yaml
mlflow:
  enabled: true
  tracking_uri: "http://<your-mlflow-server>:5000"
```

If the server is unreachable, training automatically falls back to the local
`./mlruns/` file store so training is never blocked.

---

## Checkpoint output

After training the `./attention_fusion_model/` directory contains:

```
attention_fusion_model/
├── model.pt              ← best model weights (state dict)
├── label_maps.json       ← all label→id mappings for all three tasks
├── metrics.json          ← val metrics at the time of the best checkpoint
├── tokenizer_config.json
├── vocab.txt
└── ...                   ← other tokenizer files
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

inf = AttentionFusionInference("./attention_fusion_model")

texts = [
    "I'm calling from Nairobi. My daughter Sarah, aged 12, was assaulted by her teacher.",
    "Hello, child helpline, how can I help you today?",
]

# Named Entity Recognition
ner_results = inf.predict_ner(texts)
# → [[(token, label), ...], ...]
# → e.g. [('Sarah', 'NAME'), ('12', 'AGE'), ('Nairobi', 'LOCATION'), ...]

# Case Classification
cls_results = inf.predict_classification(texts)
# → [{'main_category': 'VANE', 'sub_category': 'Physical Abuse',
#      'intervention': 'Counselling', 'priority': '1'}, ...]

# Quality Assurance scoring
qa_results = inf.predict_qa(texts)
# → [{'opening': [1], 'listening': [1,0,1,1,0],
#      'proactiveness': [0,1,0], 'resolution': [1,1,0,0,1],
#      'hold': [0,0], 'closing': [1]}, ...]
```

### Inference on GPU

```python
inf = AttentionFusionInference("./attention_fusion_model", device="cuda")
```

---

## Parameter count comparison

| Setup | Parameters | Forward passes per input |
|-------|-----------|--------------------------|
| 3 separate DistilBERT models | ~201M | 3 |
| **Attention Fusion (this model)** | **~67M** | **1** |

---

## Training strategy: task alternation

Each epoch, all batches from all three DataLoaders are collected, tagged with
their task name, and shuffled together. The model iterates through this single
shuffled list rather than processing one full task at a time.

**Why this matters:**
- The shared backbone receives gradients from all three tasks continuously,
  preventing long stretches of single-task updates.
- This reduces catastrophic forgetting compared to round-robin scheduling.
- Consistent with the approach used in MT-DNN and UniLM.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'torch'` | Use the venv: `/home/naynek/Work/ai/venv/bin/python trainer.py` |
| `CUDA out of memory` | Reduce `batch_size` to `8` or `4` in `config.yaml` |
| `MLflow connection refused` | Set `tracking_uri: "file:./mlruns"` in `config.yaml` |
| NER F1 stays near 0 for first epochs | Normal — NER is the hardest task; it typically lags CLS and QA early on |
| Early stopping triggers too soon | Increase `early_stopping_patience` from `3` to `5` in `config.yaml` |
