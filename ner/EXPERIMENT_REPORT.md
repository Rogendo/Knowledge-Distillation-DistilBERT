# Experiment Report: NER v1 - DistilBERT for Child Helpline Entities

*Date:* 2025-10-08
*AI Lead:* Rogendo
*Status:*  Completed
*Project:* OpenCHS
*Model Type:* NER

---

## Executive Summary
This report details the training and evaluation of a Named Entity Recognition (NER) model, `ner-distilbert-en-synthetic-v2`, based on the `distilbert-base-cased` architecture. The model was fine-tuned on a synthetic dataset of 2,048 samples to identify key entities in child helpline conversations. The model achieved an overall F1-score of 58.7%, showing strong performance in identifying persons (`NAME`, `VICTIM`, `PERPETRATOR`) and `AGE`, but struggling with contextual entities like `LOCATION` and `INCIDENT_TYPE`. The recommendation is to proceed with caution and focus on improving data for weaker entity types.

*Key Metrics:*
•⁠  ⁠Primary Metric (Overall F1-Score): **0.587** (Target: > 0.75)
•⁠  ⁠Status:  Partial

---

## 1. Experiment Metadata

| Field | Value |
|-------|-------|
| Model Version | v1.0.0 |
| Base Model | `distilbert-base-cased` |
| MLflow Run ID | [Link to MLflow run] |
| GitHub Branch | [Link to branch/commit] |
| Hugging Face Model | [Link if published] |
| Training Duration | ~[X hours] |
| Compute Resources | [e.g., 1x GPU, 8x CPUs] |

---

## 2. Objective & Hypothesis

### Problem Statement
Case workers at child helplines manually review conversation transcripts to extract key information like names, locations, and incident details. This process is time-consuming and prone to error. An effective NER model can automate this extraction, enabling faster case processing and better data analysis.

### Hypothesis
By fine-tuning a `distilbert-base-cased` model on a synthetic dataset of helpline conversations, we can achieve an F1-score of over 0.75 for identifying key entities, thereby significantly speeding up the information extraction process.

### Success Criteria
•⁠  ⁠*Primary Metric:* Overall Micro F1-Score > 0.75
•⁠  ⁠*Secondary Metrics:*
  - Per-entity F1-Score for `NAME`, `VICTIM`, `PERPETRATOR` > 0.80
  - Per-entity F1-Score for all other entities > 0.60
•⁠  ⁠*Qualitative Goals:* The model should correctly distinguish between different types of names (e.g., victim vs. perpetrator) and identify locations and incident types from conversational text.

---

## 3. Dataset

### Data Sources
•⁠  ⁠*Source 1:* `ner_synthetic_dataset_v2.jsonl` - 2048 samples. This is a synthetically generated dataset tailored for child helpline scenarios.

### Data Split
Based on the `config_v2.yaml` file (10% test, 10% validation).

| Split | Size | Percentage |
|-------|------|------------|
| Train | 1638 | 80% |
| Validation | 205 | 10% |
| Test | 205 | 10% |

### Data Characteristics

*For NER:*
•⁠  ⁠**Entity types (9):** `PERPETRATOR`, `LOCATION`, `LANDMARK`, `PHONE_NUMBER`, `GENDER`, `AGE`, `INCIDENT_TYPE`, `NAME`, `VICTIM`.
•⁠  ⁠**Number of entities:** Varies per sample.
•⁠  ⁠**Annotation Style:** Character-level start/end indices.

### Preprocessing Steps
1.⁠ ⁠Data loaded from JSONL file into a Hugging Face `Dataset` object.
2.⁠ ⁠Text tokenized using the `distilbert-base-cased` tokenizer.
3.⁠ ⁠Character-level entity annotations were aligned with token-level IOB2-formatted labels (`B-NAME`, `I-NAME`, etc.) for training.

### Data Ethics & Compliance
•⁠   Synthetic data used, mitigating direct privacy concerns.
•⁠   Child data protection measures considered in data generation.

---

## 4. Model Architecture & Configuration

### Base Model
•⁠  ⁠*Model:* `DistilBertForTokenClassification`
•⁠  ⁠*Source:* Hugging Face Hub (`distilbert-base-cased`)
•⁠  ⁠*Parameters:* ~65 million
•⁠  ⁠*Modifications:* A new token classification head was added and trained for the 10 specific NER labels.

### Training Configuration

*Hyperparameters (from `config_v2.yaml`):*
```yaml
learning_rate: 2.0e-05
per_device_train_batch_size: 8
per_device_eval_batch_size: 8
num_train_epochs: 5
weight_decay: 0.01
warmup_ratio: 0.1
max_sequence_length: 512
eval_strategy: "steps"
eval_steps: 1000
save_steps: 1000
load_best_model_at_end: true
metric_for_best_model: "eval_f1"
```

*Training Strategy:*
•⁠  ⁠**Approach:** Fine-tuning the pre-trained `distilbert-base-cased` model.
•⁠  ⁠**Loss function:** Cross-Entropy Loss (default for token classification).
•⁠  ⁠**Regularization:** Dropout (0.1), Weight Decay (0.01).

### Infrastructure
•⁠  ⁠*Hardware:* [GPU type and count]
•⁠  ⁠*Framework:* PyTorch
•⁠  ⁠*Key Libraries:* `transformers`, `datasets`, `torch`, `seqeval`, `scikit-learn`.

---

## 5. Results

### Quantitative Metrics

*Primary Results:*
| Metric | Baseline | Previous Best | Current | Target | Status |
|--------|----------|---------------|---------|--------|--------|
| Overall Micro F1 | N/A | N/A | **0.587** | > 0.75 | Red |

*Per-entity F1-Scores :*
| Entity | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **VICTIM** | 0.792 | 0.870 | **0.829** | 1545 |
| **NAME** | 0.855 | 0.727 | **0.786** | 6549 |
| **AGE** | 0.992 | 0.632 | **0.772** | 2046 |
| **PERPETRATOR** | 0.688 | 0.695 | **0.691** | 776 |
| **GENDER** | 0.858 | 0.460 | **0.599** | 1616 |
| **INCIDENT_TYPE** | 0.207 | 0.159 | **0.180** | 1920 |
| **LOCATION** | 0.142 | 0.223 | **0.173** | 2176 |
| **LANDMARK** | 0.000 | 0.000 | **0.000** | 155 |
| **PHONE_NUMBER**| 0.000 | 0.000 | **0.000** | 60 |

*Confusion Matrix:*
A token-level confusion matrix 

![alt text](ner_evaluation_results/ner_confusion_matrix.png)

### Qualitative Observations
•⁠  ⁠*What works well:* The model is effective at identifying entities that have strong, clear patterns, such as names of people (`NAME`, `VICTIM`), `AGE`, and `GENDER`. The high precision for `AGE` (0.992) is notable.
•⁠  ⁠*What needs improvement:* The model struggles significantly with entities that require deeper contextual understanding. `LOCATION` and `INCIDENT_TYPE` have very low scores, indicating the model confuses them with other entities or fails to identify them.
•⁠  ⁠*Unexpected findings:* The model completely failed to identify `LANDMARK` and `PHONE_NUMBER`, suggesting the patterns for these entities were not learned, possibly due to insufficient or low-quality examples in the synthetic dataset.

---

## 6. Analysis & Insights

### What Worked
1.⁠ ⁠**Transfer Learning:** Using a pre-trained DistilBERT model provided a strong foundation for learning entity patterns like names and ages, which are common in general text.
2.⁠ ⁠**Synthetic Data for Core Entities:** The synthetic data was clearly effective for training high-recall models for core entities like `VICTIM` (87% recall) and `PERPETRATOR` (69.5% recall).

### What Didn't Work
1.⁠ ⁠**Contextual Entity Recognition:** The model's architecture and the synthetic data were insufficient for learning entities that require nuanced contextual understanding, such as `INCIDENT_TYPE`. These often consist of multi-word phrases that are highly variable.
2.⁠ ⁠**Rare Entity Recognition:** The model failed to learn patterns for `LANDMARK` and `PHONE_NUMBER`. This is likely due to an insufficient number of diverse examples for these categories in the training data.

### Key Learnings
•⁠  ⁠Synthetic data is a viable strategy for bootstrapping an NER model, but it must have sufficient diversity and realism, especially for complex, contextual entities.
•⁠  ⁠Simple fine-tuning may not be enough for difficult entity types. More advanced techniques or more targeted data augmentation may be required.

---

## 7. Challenges & Limitations

### Dataset Limitations
•⁠  ⁠**Synthetic Nature:** The data, while tailored, may lack the complexity, noise, and diversity of real-world helpline conversations.
•⁠  ⁠**Imbalance:** The support counts show a large number of `NAME` entities compared to `LANDMARK` or `PHONE_NUMBER`, creating a class imbalance that likely contributed to the poor performance on rare entities.

### Model Limitations
•⁠  ⁠**Contextual Understanding:** DistilBERT, while efficient, may have a limited capacity to understand the deep, situational context required for entities like `INCIDENT_TYPE` compared to larger models (e.g., RoBERTa-Large).

---

## 8. Reproducibility

### Environment Setup
```bash
# Clone repository and navigate to the task directory
# git clone ...
# cd .../tasks/ner

# Install dependencies
pip install -r requirements.txt 
# (Note: A requirements.txt was not present, but key libraries include torch, transformers, datasets, seqeval, scikit-learn, pyyaml)
```

### Training Command
```bash
# Command to reproduce this experiment using the trainer script
python trainer.py --config config_v2.yaml
```

### Evaluation Command
```bash
# Command to evaluate the model
python evaluate_ner.py
```

---

## 9. Next Steps & Recommendations

### Immediate Actions
•⁠  ⁠[ ] **Analyze Failure Cases:** Manually review the errors for `LOCATION` and `INCIDENT_TYPE` to identify common failure patterns.
•⁠  ⁠[ ] **Augment Data:** Generate or source more diverse and realistic training examples for the low-performing categories (`INCIDENT_TYPE`, `LOCATION`, `LANDMARK`, `PHONE_NUMBER`).
•⁠  ⁠[ ] **Error Analysis on Confusion Matrix:** Deeply analyze the `ner_confusion_matrix.png` to see which specific labels are being confused.

### Future Experiments
1.⁠ ⁠*Hypothesis:* Using a larger base model (e.g., `bert-base-cased` or `roberta-base`) will improve performance on contextual entities.
   - *Approach:* Re-run the training pipeline with a different base model.
   - *Expected impact:* Increase in F1-score for `INCIDENT_TYPE` and `LOCATION` by at least 10 points.

### Decision Point
*Recommendation:*
•⁠  ⁠[X]  ***Continue iteration*** - Run additional experiments.

*Justification:* The model shows promise but does not meet the success criteria for deployment. The poor performance on critical entities like `LOCATION` and `INCIDENT_TYPE` makes it unreliable for fully automated use. The immediate next step should be data augmentation and possibly experimenting with a larger model.
