# Knowledge Distillation & Attention Fusion with DistilBERT

A multi-task NLP pipeline for child helpline conversation analysis. The system trains a single shared **DistilBERT** backbone across three distinct NLP tasks — Named Entity Recognition, Case Classification, and Quality Assurance scoring — using an attention-fusion architecture with task-alternation training.

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Knowledge Distillation](#2-knowledge-distillation)
3. [Repository Structure](#3-repository-structure)
4. [Data](#4-data)
5. [Single-Task Models](#5-single-task-models)
6. [Attention Fusion — Multi-Task Architecture](#6-attention-fusion--multi-task-architecture)
7. [Task-Alternation Training Strategy](#7-task-alternation-training-strategy)
8. [Training Configuration](#8-training-configuration)
9. [Running the Pipeline](#9-running-the-pipeline)
10. [MLflow Experiment Tracking](#10-mlflow-experiment-tracking)
11. [Inference](#11-inference)
12. [Label Reference](#12-label-reference)

---

## 1. Problem Definition

Child helpline conversations contain rich, multi-layered information that requires several distinct types of analysis to be operationally useful:

| Task | Type | What it answers |
|------|------|-----------------|
| **Named Entity Recognition (NER)** | Token-level | *Who* is mentioned in the call? Names, ages, locations, incident types, perpetrators |
| **Case Classification** | Sentence-level | *What* is the case about? Category, sub-category, required intervention, and urgency priority |
| **Quality Assurance (QA) Scoring** | Sentence-level | *How well* did the counselor handle the call? 6 behavioural dimensions, 17 binary sub-metrics |

Training three separate models for these tasks is expensive: each full DistilBERT checkpoint is ~66M parameters. Because all three tasks read the same type of input — a conversation transcript — their representations overlap significantly. The goal of this project is to:

1. Share a single backbone encoder across all three tasks.
2. Give each task its own learned attention pooling head so it can focus on the tokens most relevant to its objective.
3. Train the shared backbone with gradient signal from all three tasks simultaneously.

---

## 2. Knowledge Distillation

### What is DistilBERT?

DistilBERT (Sanh et al., 2019) is itself the product of **knowledge distillation** from BERT. During distillation, a smaller *student* model is trained to mimic the outputs of a larger *teacher* model (BERT-base, 110M parameters) using a combination of:

- **Soft-target loss** — KL divergence between teacher and student output distributions (temperature-scaled softmax)
- **Cosine embedding loss** — alignment of hidden states between teacher and student layers
- **Masked language model loss** — the student's own task objective

The result is DistilBERT: 40% fewer parameters (66M), 60% faster inference, and 97% of BERT's performance on GLUE benchmarks. This project inherits those compression benefits as a foundation.

### Task-Level Knowledge Sharing

Within this project, knowledge sharing operates at the **task level** through the shared encoder:

```
Teacher signals from all three tasks flow into the same backbone weights.
Each task provides gradients that push the shared DistilBERT representations
toward being useful for NER, classification, and QA simultaneously.
```

This is a form of **multi-task learning as implicit knowledge transfer** — the backbone learns features that are generalisable across all three tasks rather than over-specialising to any one.

### Future Extension: Explicit KD

An explicit distillation phase is planned where:
1. Three task-specific teacher models (NER, CLS, QA) are trained to convergence separately.
2. The attention fusion student is trained with KL-divergence loss against each teacher's soft logits, in addition to the standard hard-label task losses.

---

## 3. Repository Structure

```
Knowledge-Distillation-DistilBERT/
│
├── data/                                     # Raw training data (all tasks)
│   ├── ner_synthetic_dataset_v1.jsonl        # 2,000 NER records (JSONL)
│   ├── cases_generated_data_v0005.json       # 10,000 case records (JSON array)
│   ├── synthetic_qa_metrics_data_v01x.json   # 4,996 QA records (JSON array)
│   └── balanced_cases_generated_data_v0005.json
│
├── ner/                                      # Single-task NER trainer
│   ├── trainer.py                            # HF AutoModelForTokenClassification
│   ├── config_v2.yaml
│   ├── evaluate_ner.py
│   ├── inference.py
│   └── EXPERIMENT_REPORT.md
│
├── classification/                           # Single-task Case Classification trainer
│   ├── trainer.py                            # Custom MultiTaskDistilBert model
│   ├── config.yaml
│   ├── eval.py
│   └── test.py
│
├── quality_assurance/                        # Single-task QA Scoring trainer
│   ├── train.py                              # MultiHeadQAClassifier model
│   ├── eval.py
│   ├── inference.py
│   └── config.yaml
│
└── attention_fusion/                         # Multi-task Attention Fusion model
    ├── model.py                              # AttentionFusionModel + TaskAttentionPooling
    ├── datasets.py                           # NERDataset, ClassificationDataset, QADataset
    ├── trainer.py                            # MultiTaskTrainer (task-alternation)
    ├── inference.py                          # AttentionFusionInference
    └── config.yaml
```

---

## 4. Data

All datasets are synthetically generated to represent child helpline calls in an East African context (English/Swahili code-switching, Kenyan county geography, local incident categories).

### 4.1 NER Data — `ner_synthetic_dataset_v1.jsonl`

**Format**: JSONL (one JSON object per line)
**Size**: 2,000 records
**Split**: 90% train / 10% val (NER records exceeding 512 tokens are filtered out, not truncated — truncation would silently drop entity labels)

**Schema**:
```json
{
  "text": "Hello, is this the child helpline? Yes, I'm Mwakamba calling from Kisumu...",
  "entities": [
    {"start": 44, "end": 52,  "label": "NAME"},
    {"start": 65, "end": 71,  "label": "LOCATION"},
    {"start": 97, "end": 113, "label": "VICTIM"},
    {"start": 151,"end": 154, "label": "AGE"},
    {"start": 241,"end": 259, "label": "INCIDENT_TYPE"}
  ]
}
```

- `text` — raw conversation transcript
- `entities` — list of character-level spans with an entity label
- Spans use half-open intervals `[start, end)` matching Python slicing

**Entity Labels (10 classes)**:

| Label | Description |
|-------|-------------|
| `O` | Outside any entity (background) |
| `NAME` | Name of caller, child, or person mentioned |
| `LOCATION` | County, town, landmark, or area |
| `VICTIM` | The child at risk |
| `AGE` | Age of victim or perpetrator |
| `GENDER` | Gender reference |
| `INCIDENT_TYPE` | Nature of the abuse or issue |
| `PERPETRATOR` | Person responsible for harm |
| `PHONE_NUMBER` | Contact numbers mentioned |
| `LANDMARK` | Specific named places |

**Token-label alignment**: Offset mapping is used to align character-level spans to sub-word tokens. Only the first sub-word of a multi-token word is assigned the entity label; continuation tokens and special/padding tokens receive `-100` (ignored in loss).

---

### 4.2 Classification Data — `cases_generated_data_v0005.json`

**Format**: JSON array
**Size**: 9,996 records
**Split**: 90% train / 10% val (long narratives are truncated to 512 tokens — labels are sentence-level so truncation is safe)

**Schema**:
```json
{
  "uniqueid": "1696525048.1374",
  "startdate": "05 Oct 2023",
  "starttime": "19:57:28",
  "category": "Drug/Alcohol Abuse",
  "victim": {"gender": "female", "first_name": "Eunice", "age": "5"},
  "reporter": {"gender": "female", "relationship": "Parent"},
  "perpetrator": {"gender": "female", "age": "Middle-aged (41-60)"},
  "county": "Murang'A",
  "counselor": "Counselor N",
  "intervention": "Counselling",
  "priority": "1",
  "narrative": "Rose Wakesho (Caller): Hello, I need help. My daughter..."
}
```

- `narrative` — the input text fed to the model (full call transcript)
- `category` — one of 77 sub-categories (the primary classification target)
- `intervention` — one of 16 intervention types
- `priority` — `"1"` (critical), `"2"` (urgent), or `"3"` (routine)

**The 4 classification heads**:

| Head | Target field | Classes |
|------|-------------|---------|
| Main Category | `category` → mapped via `SUB_TO_MAIN` | 8 |
| Sub-Category | `category` | 77 |
| Intervention | `intervention` | 16 |
| Priority | `priority` | 3 |

**Main Category mapping** (8 classes): Sub-categories are grouped into 8 main categories via a hardcoded `SUB_TO_MAIN` dictionary in `datasets.py`:

```
Advice and Counselling  |  Child Maintenance & Custody  |  Disability
GBV                     |  Information                  |  Nutrition
VANE                    |  Unknown
```

---

### 4.3 QA Scoring Data — `synthetic_qa_metrics_data_v01x.json`

**Format**: JSON array
**Size**: 4,996 records
**Split**: 90% train / 10% val

**Schema**:
```json
{
  "text": "Hey there, you've reached the Child Helpline, whatcha got?...",
  "labels": "{\"opening\": [0], \"listening\": [1, 0, 0, 0, 0], \"proactiveness\": [1, 0, 0], \"resolution\": [1, 0, 0, 0, 0], \"hold\": [0, 0], \"closing\": [0]}",
  "sample_id": "qa_1",
  "scenario": "child_abuse_report",
  "quality_level": "poor"
}
```

- `text` — full call transcript (the model input)
- `labels` — a **JSON string** encoding 6 behavioural dimension heads. Each head contains a binary list (1 = criterion met, 0 = not met)
- `quality_level` — metadata label (`"poor"`, `"average"`, `"good"`) — not used in training

**The 6 QA heads and their 17 sub-metrics**:

| Head | # Sub-metrics | Sub-metric labels |
|------|:---:|---|
| `opening` | 1 | Use of call opening phrase |
| `listening` | 5 | Caller not interrupted · Empathises · Paraphrases issue · Uses please/thank you · Doesn't hesitate |
| `proactiveness` | 3 | Willing to solve extra issues · Confirms satisfaction · Follows up on updates |
| `resolution` | 5 | Accurate information · Correct language · Consults if unsure · Follows correct steps · Explains solution |
| `hold` | 2 | Explains before placing on hold · Thanks caller for holding |
| `closing` | 1 | Proper call closing phrase used |

**Data quirks handled by the pipeline**:

- `labels` is stored as a **JSON string** — parsed with `json.loads()` at load time
- Some records contain **variable-length label lists** (e.g. `listening` may have 5 or 6 values) — truncated/padded to the expected size
- Some `hold` labels contain **non-numeric sentinels** such as `"No hold was required"` — coerced to `0.0` (head not applicable for that call)

---

## 5. Single-Task Models

Each task has a dedicated single-task trainer used for baseline comparison and independent iteration.

### 5.1 NER — `ner/trainer.py`

**Model**: `AutoModelForTokenClassification` from HuggingFace Transformers wrapping `distilbert-base-cased`
**Framework**: HuggingFace `Trainer` with `TrainingArguments`

```
distilbert-base-cased
       ↓ last_hidden_state [batch, seq_len, 768]
  Linear(768 → 10)
       ↓ token logits [batch, seq_len, 10]
  CrossEntropyLoss(ignore_index=-100)
```

**Key settings**:
- `max_length`: 512 tokens
- `batch_size`: 8
- `epochs`: 5
- `eval_strategy`: steps, metric: `eval_f1`
- `load_best_model_at_end`: true

**Results** (see `ner/EXPERIMENT_REPORT.md`):

| Entity | F1 |
|--------|----|
| VICTIM | 0.829 |
| NAME | 0.786 |
| AGE | 0.772 |
| Overall Micro F1 | **0.587** |
| LOCATION | 0.173 |
| LANDMARK | 0.000 |
| PHONE_NUMBER | 0.000 |

Overall F1 of 0.587 is below the target of 0.75. Rare and contextually ambiguous entity types (LANDMARK, PHONE_NUMBER, LOCATION) drive the shortfall. The multi-task fusion model targets improvement by incorporating richer task supervision signals into the shared backbone.

---

### 5.2 Case Classification — `classification/trainer.py`

**Model**: Custom `MultiTaskDistilBert`

```
distilbert-base-uncased
       ↓ [CLS] token [batch, 768]
  Linear(768 → 768) + ReLU + Dropout
       ↓
  ┌─────────┬──────────┬─────────────┬──────────┐
main(→8)  sub(→77)  interv(→16)  priority(→3)
  └─────────┴──────────┴─────────────┴──────────┘
  CrossEntropyLoss × 4 (summed)
```

**Key differences from attention fusion**:
- Uses fixed `[CLS]` pooling — no learned task-specific attention
- Uses `distilbert-base-uncased` (lowercased vocabulary)
- Pre-classifier projection layer (768→768) for non-linear feature transformation

**Training**: Custom loop with MLflow logging, stratified train/test split by sub-category, early stopping on sum of 4 validation losses.

---

### 5.3 QA Scoring — `quality_assurance/train.py`

**Model**: Custom `MultiHeadQAClassifier`

```
distilbert-base-uncased
       ↓ [CLS] token [batch, 768]
  Dropout
       ↓
  ┌──────────┬───────────┬────────────────┬────────────┬────────┬─────────┐
opening(→1) listen(→5) proactiv(→3) resolut(→5)  hold(→2) closing(→1)
  └──────────┴───────────┴────────────────┴────────────┴────────┴─────────┘
  BCEWithLogitsLoss × 6 (summed)
```

**Key settings**:
- `max_length`: 256 tokens (shorter than other tasks — QA focuses on the interaction pattern, not the full transcript)
- `batch_size`: 4
- Sigmoid activation at inference time (not in forward pass — raw logits used for loss)

**Training**: Custom loop with per-epoch validation metrics (accuracy, precision, recall, F1 per head), early stopping, best-model checkpoint save.

---

## 6. Attention Fusion — Multi-Task Architecture

The attention fusion model eliminates three separate DistilBERT instances and replaces them with **one shared backbone** and **task-specific output heads**.

### 6.1 Overall Architecture

```
                        Input Text
                            │
                        Tokenizer
                  [input_ids, attention_mask]
                            │
               ┌────────────────────────────┐
               │   DistilBERT Backbone      │
               │   distilbert-base-cased    │
               │   66M parameters           │
               │   → last_hidden_state      │
               │     [batch, seq_len, 768]  │
               └─────────────┬──────────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
     task='ner'        task='cls'       task='qa'
            │                │                │
   ┌────────────────┐  ┌─────────────┐  ┌─────────────┐
   │  NER Head      │  │  CLS Head   │  │   QA Head   │
   │                │  │             │  │             │
   │ Linear(768→768)│  │TaskAttn     │  │ TaskAttn    │
   │  + GELU        │  │Pooling      │  │ Pooling     │
   │  + Dropout     │  │[batch,768]  │  │ [batch,768] │
   │                │  │  Dropout    │  │   Dropout   │
   │ Linear(768→10) │  │             │  │             │
   │ [batch,seq,10] │  │ 4 heads:    │  │ 6 heads:    │
   │                │  │ main  (→8)  │  │ open  (→1)  │
   │  CE Loss       │  │ sub   (→77) │  │ listen(→5)  │
   │ (ignore=-100)  │  │ interv(→16) │  │ proact(→3)  │
   └────────────────┘  │ prior (→3)  │  │ resolv(→5)  │
                       │             │  │ hold  (→2)  │
                       │ CE Loss × 4 │  │ close (→1)  │
                       └─────────────┘  │ BCE Loss × 6│
                                        └─────────────┘
```

**Parameter efficiency**: 3 separate models = ~200M parameters. One attention fusion model = ~66M parameters + small head weights.

---

### 6.2 Task Attention Pooling

All single-task baselines use the `[CLS]` token as the sentence representation. The attention fusion model replaces this with a **learned attention pooling layer per task** (`TaskAttentionPooling` in `model.py`).

```python
class TaskAttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        self.attn = nn.Linear(hidden_size, 1)   # one scalar score per token

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch, seq_len, 768]
        scores = self.attn(hidden_states)        # [batch, seq_len, 1]

        # Mask padding tokens before softmax
        scores = scores.masked_fill(
            attention_mask.unsqueeze(-1) == 0, float('-inf')
        )
        weights = torch.softmax(scores, dim=1)   # [batch, seq_len, 1]

        # Weighted sum over sequence dimension
        pooled = (weights * hidden_states).sum(dim=1)  # [batch, 768]
        return pooled
```

**Why this matters**: The NER task cares about every individual token. The classification task cares about the overall case type — which may be signalled by just a few key phrases in a long narrative. The QA task cares about specific behavioural moments (greeting, hold phrase, closing). A fixed `[CLS]` token forces all tasks to share the same pooling point. Per-task attention pooling lets each head learn to aggregate tokens that are most informative for its objective.

---

### 6.3 Model Routing

A single `forward()` call routes computation through the appropriate head based on the `task` argument:

```python
def forward(self, input_ids, attention_mask, task: str, labels=None):
    # Shared backbone — always runs
    hidden = self.backbone(input_ids, attention_mask).last_hidden_state

    if task == 'ner':
        return self._forward_ner(hidden, labels)
    elif task == 'cls':
        return self._forward_cls(hidden, attention_mask, labels)
    elif task == 'qa':
        return self._forward_qa(hidden, attention_mask, labels)
```

Only the backbone and the relevant head run for each batch. The other heads receive no gradients for that step.

---

### 6.4 Loss Functions

| Task | Loss | Details |
|------|------|---------|
| NER | `CrossEntropyLoss(ignore_index=-100)` | Token-level; `-100` masks padding and special tokens |
| CLS | `CrossEntropyLoss(ignore_index=-1) × 4` | One loss per head, summed; `-1` for unlabelled samples |
| QA | `BCEWithLogitsLoss() × 6` | Binary multi-label; one loss per head, summed |

---

### 6.5 Shared Encoder Advantages

| Property | Separate Models | Attention Fusion |
|----------|:--------------:|:----------------:|
| Parameters | ~200M (3 × 66M) | ~66M |
| NER generalisation | NER gradients only | NER + CLS + QA gradients |
| Inference deployments | 3 | 1 |
| Cross-task regularisation | None | Implicit through shared weights |
| Task-specific representation | Fixed [CLS] | Learned attention pooling |

---

## 7. Task-Alternation Training Strategy

### 7.1 The Problem with Round-Robin

A naive approach to multi-task training is to train one task per epoch in sequence (NER epoch → CLS epoch → QA epoch → repeat). This causes **catastrophic forgetting**: when the model trains exclusively on CLS data, it overwrites NER-specific patterns in the shared backbone. By the time QA training begins, the backbone has drifted from what NER needs.

### 7.2 Task Alternation: Interleaved Batch Shuffling

The attention fusion trainer uses a different strategy: at the start of every epoch, all batches from all three DataLoaders are collected into a single list, each tagged with its task name, and then the entire list is randomly shuffled.

```python
def train_epoch(self):
    # Step 1: Tag every batch with its task
    all_batches = []
    for task, loader in self.train_loaders.items():
        for batch in loader:
            all_batches.append((task, batch))

    # Step 2: Shuffle all batches together (task interleaving)
    random.shuffle(all_batches)

    # Step 3: Iterate through the shuffled list
    for task, batch in all_batches:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels']  # structure depends on task

        output = model(input_ids, attention_mask, task=task, labels=labels)
        loss   = output['loss']

        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

### 7.3 Why This Works

- **Continuous multi-task gradient signal**: The backbone receives gradients from NER, CLS, and QA in every epoch, interleaved randomly. No task is ever "frozen out" for an entire epoch.
- **Prevents catastrophic forgetting**: Because a QA batch might be immediately followed by a NER batch, the backbone cannot drift far from any task's requirements.
- **Better backbone regularisation**: The shared weights must satisfy constraints from all three tasks simultaneously, which acts as a regulariser and reduces overfitting to any single task.
- **Dataset size imbalance handling**: With ~9K CLS, ~5K QA, and ~2K NER batches per epoch, random shuffling naturally respects the dataset proportions without requiring manual weighting.

### 7.4 Batch Counts per Epoch

With `batch_size=16` and `test_size=0.1`:

| Task | Train records | Batches/epoch |
|------|:---:|:---:|
| NER | ~1,798 | ~112 |
| CLS | ~8,996 | ~562 |
| QA | ~4,496 | ~281 |
| **Total** | **~15,290** | **~955** |

All ~955 batches are shuffled into a single sequence each epoch.

### 7.5 Learning Rate Schedule

A single `get_linear_schedule_with_warmup` scheduler governs the entire training run. Total steps and warmup steps are computed across all tasks:

```python
total_batches = sum(len(dl) for dl in train_loaders.values())  # ~955
total_steps   = total_batches * num_epochs                     # ~9,550 for 10 epochs
warmup_steps  = int(total_steps * warmup_ratio)                # ~955 (10%)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

The learning rate linearly warms up for the first 10% of steps, then linearly decays to zero.

### 7.6 Validation and Early Stopping

After each epoch, the model is evaluated separately on each task's validation set:

```python
# NER validation: macro F1 over non-padding tokens
ner_f1  = mean([macro_f1(logits, labels, ignore=-100) for batch in val_ner])

# CLS validation: average accuracy across 4 heads
cls_acc = mean([avg_head_accuracy(logits, labels) for batch in val_cls])

# QA validation: average micro-F1 across 6 heads at threshold=0.5
qa_f1   = mean([avg_head_f1(logits, labels) for batch in val_qa])

# Composite score for early stopping and checkpoint selection
avg_val_metric = (ner_f1 + cls_acc + qa_f1) / 3
```

If `avg_val_metric` does not improve for `patience=3` consecutive epochs, training stops and the best checkpoint is retained.

---

## 8. Training Configuration

### Attention Fusion (`attention_fusion/config.yaml`)

```yaml
backbone:
  model_name: "distilbert-base-cased"   # Cased — NER is case-sensitive
  dropout: 0.1

data:
  ner_path:             "../data/ner_synthetic_dataset_v1.jsonl"
  classification_path:  "../data/cases_generated_data_v0005.json"
  qa_path:              "../data/synthetic_qa_metrics_data_v01x.json"
  test_size: 0.1
  random_seed: 42

tokenizer:
  max_length: 512       # DistilBERT hard limit
  padding:    "max_length"
  truncation: true
  # NER records exceeding max_length are FILTERED OUT (not truncated)
  # because truncation would silently drop tail entity labels.
  # CLS and QA records are truncated safely — labels are sentence-level.

training:
  num_epochs:               10
  batch_size:               16
  learning_rate:            2.0e-5
  weight_decay:             0.01
  warmup_ratio:             0.1
  max_grad_norm:            1.0
  early_stopping_patience:  3

output:
  model_dir:        "./attention_fusion_model"
  label_maps_file:  "label_maps.json"
  metrics_file:     "metrics.json"

mlflow:
  enabled:          true
  tracking_uri:     "file:./mlruns"
  experiment_name:  "attention_fusion_multitask"
  run_name:         "multitask_train"
```

### Why `distilbert-base-cased`?

NER is case-sensitive — `"John"` (a name) versus `"john"` (part of a sentence) carry different information. Using the cased model preserves capitalisation signals for the NER head. The CLS and QA tasks are less sensitive to case, but share the backbone for consistency.

---

## 9. Running the Pipeline

### Prerequisites

```bash
python -m venv venv && source venv/bin/activate
pip install torch transformers scikit-learn pandas pyyaml mlflow tqdm
```

### Single-Task: NER

```bash
cd ner/
python trainer.py        # uses config_v2.yaml by default
python evaluate_ner.py   # generates confusion matrix and metrics
```

### Single-Task: Case Classification

```bash
cd classification/
python trainer.py        # uses config.yaml
python eval.py
```

### Single-Task: QA Scoring

```bash
cd quality_assurance/
python train.py          # uses config.yaml
python eval.py           # generates per-head confusion matrices
```

### Attention Fusion (Multi-Task)

```bash
cd attention_fusion/
python trainer.py --config config.yaml
```

**Expected output**:
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
Training on cuda
Epochs: 10, patience: 3

Epoch 1/10
  Training: 100%|████| 955/955 ...
  train_loss=2.3141  ner=0.8832  cls=0.9502  qa=0.4908
  val  — NER F1: 0.412  CLS acc: 0.531  QA F1: 0.483  avg: 0.475
  [Checkpoint saved]
```

---

## 10. MLflow Experiment Tracking

The attention fusion trainer integrates with MLflow for full experiment reproducibility.

**Start the MLflow UI** (local file store):
```bash
cd attention_fusion/
mlflow ui --backend-store-uri file:./mlruns
# Open http://localhost:5000
```

**Or point to a remote server** in `config.yaml`:
```yaml
mlflow:
  tracking_uri: "http://192.168.8.18:5000"
```

**Tracked per run**:

| Category | Items logged |
|----------|-------------|
| Parameters | All hyperparameters, data filter stats (% kept per task) |
| Tags | Device, total parameter count, training strategy (`task_alternation`) |
| Metrics (per epoch) | `train/loss`, `train/loss_ner`, `train/loss_cls`, `train/loss_qa`, `val/ner_f1`, `val/cls_avg_acc`, `val/qa_avg_f1`, `val/avg` |
| Artifacts | `model.pt`, `label_maps.json`, `metrics.json` (best checkpoint only) |

---

## 11. Inference

Load the saved checkpoint and run all three tasks through a single model object:

```python
from attention_fusion.inference import AttentionFusionInference

inf = AttentionFusionInference(
    model_dir="./attention_fusion/attention_fusion_model",
    device="cuda",
    qa_threshold=0.5,
)

texts = [
    "Hello, I'm calling because my 8-year-old son John in Nakuru is being beaten by his stepfather..."
]

# Named Entity Recognition
ner_results = inf.predict_ner(texts)
# → [[("Hello", "O"), (",", "O"), ("John", "NAME"), ("Nakuru", "LOCATION"), ...]]

# Case Classification
cls_results = inf.predict_classification(texts)
# → [{"main_category": "GBV", "sub_category": "Physical Abuse",
#     "intervention": "Referral to Social Services", "priority": "1"}]

# Quality Assurance Scoring
qa_results = inf.predict_qa(texts)
# → [{"opening": [1], "listening": [1, 0, 1, 1, 0], "proactiveness": [0, 1, 0],
#     "resolution": [1, 1, 0, 1, 1], "hold": [0, 0], "closing": [1]}]
```

---

## 12. Label Reference

### NER Labels

```
O  NAME  LOCATION  VICTIM  AGE  GENDER  INCIDENT_TYPE  PERPETRATOR  PHONE_NUMBER  LANDMARK
```

### Classification: Main Categories (8)

```
Advice and Counselling  |  Child Maintenance & Custody  |  Disability
GBV                     |  Information                  |  Nutrition
VANE                    |  Unknown
```

### Classification: Interventions (16)

Counselling · Referral · Follow-up · Legal Aid · Psychosocial Support · Medical Referral · Home Visit · Case Conference · Police Referral · Child Protection · Emergency Response · Mediation · Shelter · Nutritional Support · Educational Support · Unknown

### Classification: Priority

| Value | Meaning |
|:-----:|---------|
| `1` | Critical — immediate risk to child |
| `2` | Urgent — action required within 24h |
| `3` | Routine — standard case management |

### QA Head Sub-metrics

| Head | Sub-metric labels |
|------|------------------|
| `opening` (1) | Use of call opening phrase |
| `listening` (5) | Caller not interrupted · Empathises with caller · Paraphrases/rephrases issue · Uses please and thank you · Does not hesitate or sound unsure |
| `proactiveness` (3) | Willing to solve extra issues · Confirms satisfaction with action points · Follows up on case updates |
| `resolution` (5) | Gives accurate information · Correct language use · Consults if unsure · Follows correct steps · Explains solution process clearly |
| `hold` (2) | Explains before placing on hold · Thanks caller for holding |
| `closing` (1) | Proper call closing phrase used |
