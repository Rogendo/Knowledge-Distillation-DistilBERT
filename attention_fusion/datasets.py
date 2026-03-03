"""
Datasets and data module for Attention Fusion Multi-Task Model.

Three task-specific datasets share a single tokenizer but produce
differently structured label tensors:
  - NERDataset          → token-level label sequence
  - ClassificationDataset → dict of 4 sentence-level integer labels
  - QADataset           → dict of 6 float tensors (binary multi-label)

MultiTaskDataModule wraps all three and exposes interleaved DataLoaders.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NER_LABELS = [
    'O', 'NAME', 'LOCATION', 'VICTIM', 'AGE', 'GENDER',
    'INCIDENT_TYPE', 'PERPETRATOR', 'PHONE_NUMBER', 'LANDMARK',
]

# Sub-category → main-category mapping (from classification/trainer.py)
SUB_TO_MAIN = {
    "Bullying": "Advice and Counselling",
    "Child in Conflict with the Law": "Advice and Counselling",
    "Discrimination": "Advice and Counselling",
    "Drug/Alcohol Abuse": "Advice and Counselling",
    "Family Relationship": "Advice and Counselling",
    "HIV/AIDS": "Advice and Counselling",
    "Homelessness": "Advice and Counselling",
    "Legal issues": "Advice and Counselling",
    "Missing Child": "Advice and Counselling",
    "Peer Relationships": "Advice and Counselling",
    "Physical Health": "Advice and Counselling",
    "Psychosocial/Mental Health": "Advice and Counselling",
    "Relationships (Boy/Girl)": "Advice and Counselling",
    "Relationships (Parent/Child)": "Advice and Counselling",
    "Relationships (Student/Teacher)": "Advice and Counselling",
    "School related issues": "Advice and Counselling",
    "Self Esteem": "Advice and Counselling",
    "Sexual & Reproductive Health": "Advice and Counselling",
    "Student/ Teacher Relationship": "Advice and Counselling",
    "Teen Pregnancy": "Advice and Counselling",
    "Adoption": "Child Maintenance & Custody",
    "Birth Registration": "Child Maintenance & Custody",
    "Custody": "Child Maintenance & Custody",
    "Foster Care": "Child Maintenance & Custody",
    "Maintenance": "Child Maintenance & Custody",
    "No Care Giver": "Child Maintenance & Custody",
    "Other": "Child Maintenance & Custody",
    "Albinism": "Disability",
    "Hearing impairment": "Disability",
    "Hydrocephalus": "Disability",
    "Mental impairment": "Disability",
    "Multiple disabilities": "Disability",
    "Physical impairment": "Disability",
    "Speech impairment": "Disability",
    "Spinal bifida": "Disability",
    "Visual impairment": "Disability",
    "Emotional/Psychological Violence": "GBV",
    "Financial/Economic Violence": "GBV",
    "Forced Marriage Violence": "GBV",
    "Harmful Practice": "GBV",
    "Physical Violence": "GBV",
    "Sexual Violence": "GBV",
    "Child Abuse": "Information",
    "Child Rights": "Information",
    "Info on Helpline": "Information",
    "Legal Issues": "Information",
    "School Related Issues": "Information",
    "Balanced Diet": "Nutrition",
    "Breastfeeding": "Nutrition",
    "Feeding & Food preparation": "Nutrition",
    "Malnutrition": "Nutrition",
    "Obesity": "Nutrition",
    "Stagnation": "Nutrition",
    "Underweight": "Nutrition",
    "Child Abduction": "VANE",
    "Child Labor": "VANE",
    "Child Marriage": "VANE",
    "Child Neglect": "VANE",
    "Child Trafficking": "VANE",
    "Emotional Abuse": "VANE",
    "Female Genital Mutilation": "VANE",
    "OCSEA": "VANE",
    "Physical Abuse": "VANE",
    "Sexual Abuse": "VANE",
    "Traditional Practice": "VANE",
    "Unlawful Confinement": "VANE",
}

# QA head config: head_name → number of binary output neurons
QA_HEAD_CONFIG: Dict[str, int] = {
    "opening": 1,
    "listening": 5,
    "proactiveness": 3,
    "resolution": 5,
    "hold": 2,
    "closing": 1,
}


# ---------------------------------------------------------------------------
# NER Dataset
# ---------------------------------------------------------------------------

class NERDataset(Dataset):
    """
    Reads NER records from a JSONL file.

    Each record: {"text": "...", "entities": [{"start": int, "end": int, "label": str}, ...]}

    Token-label alignment re-implements the offset-mapping approach from
    ner/trainer.py:69-107, using -100 to mask special and padding tokens.
    """

    def __init__(
        self,
        records: list,
        tokenizer,
        label_to_id: Dict[str, int],
        max_length: int = 512,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        text = record['text']
        entities = record.get('entities', [])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        offset_mapping = encoding['offset_mapping'].squeeze(0).tolist()

        # Build label sequence:
        #   -100  → special tokens ([CLS]/[SEP]) and padding (ignored by CrossEntropy)
        #   label → entity token
        #   O id  → non-entity real token
        labels = []
        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            if attention_mask[i] == 0:
                # Padding token
                labels.append(-100)
            elif tok_start == 0 and tok_end == 0:
                # Special token ([CLS] or [SEP])
                labels.append(-100)
            else:
                labels.append(self.label_to_id['O'])

        # Overlay entity labels on top of the default O labels
        for entity in entities:
            ent_start = entity['start']
            ent_end = entity['end']
            ent_label = self.label_to_id.get(entity['label'], self.label_to_id['O'])
            for i, (tok_start, tok_end) in enumerate(offset_mapping):
                if labels[i] == -100:
                    continue
                if tok_start >= ent_start and tok_end <= ent_end:
                    labels[i] = ent_label

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Classification Dataset
# ---------------------------------------------------------------------------

class ClassificationDataset(Dataset):
    """
    Reads case records from a JSON file.

    Each record has: narrative (text), category (sub-cat), intervention, priority.
    The sub→main mapping is applied using SUB_TO_MAIN.
    Labels are integer IDs looked up from label_maps.
    """

    def __init__(
        self,
        records: list,
        tokenizer,
        label_maps: dict,
        max_length: int = 512,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label_maps = label_maps
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        text = str(record.get('narrative') or '')

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )

        sub_cat = record.get('category', '')
        main_cat = SUB_TO_MAIN.get(sub_cat, 'Unknown')

        priority_raw = str(record.get('priority', ''))

        labels = {
            'main_category_id': torch.tensor(
                self.label_maps['main_cat2id'].get(main_cat, -1), dtype=torch.long
            ),
            'sub_category_id': torch.tensor(
                self.label_maps['sub_cat2id'].get(sub_cat, -1), dtype=torch.long
            ),
            'intervention_id': torch.tensor(
                self.label_maps['interv2id'].get(record.get('intervention', ''), -1),
                dtype=torch.long,
            ),
            'priority_id': torch.tensor(
                self.label_maps['priority2id'].get(priority_raw, -1), dtype=torch.long
            ),
        }

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
        }


# ---------------------------------------------------------------------------
# QA Dataset
# ---------------------------------------------------------------------------

class QADataset(Dataset):
    """
    Reads QA records from a JSON file.

    Each record: {"text": "...", "labels": "{...JSON string...}"}
    The labels field is a JSON string with 6 binary-list heads.
    """

    def __init__(self, records: list, tokenizer, max_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        text = record['text']

        # labels may be stored as a JSON string
        raw_labels = record['labels']
        if isinstance(raw_labels, str):
            labels_dict = json.loads(raw_labels)
        else:
            labels_dict = raw_labels

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )

        label_tensors = {
            head: torch.tensor(labels_dict[head], dtype=torch.float)
            for head in QA_HEAD_CONFIG
        }

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label_tensors,
        }


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def _ner_collate(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
    }


def _cls_collate(batch):
    label_keys = batch[0]['labels'].keys()
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels': {k: torch.stack([b['labels'][k] for b in batch]) for k in label_keys},
    }


def _qa_collate(batch):
    label_keys = batch[0]['labels'].keys()
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels': {k: torch.stack([b['labels'][k] for b in batch]) for k in label_keys},
    }


# ---------------------------------------------------------------------------
# Token-length filtering
# ---------------------------------------------------------------------------

def count_tokens(text: str, tokenizer) -> int:
    """Return the number of tokens produced by the tokenizer for a text string."""
    return len(tokenizer.encode(text, add_special_tokens=True))


def filter_by_token_length(
    records: list,
    tokenizer,
    text_field: str,
    max_length: int,
) -> Tuple[list, dict]:
    """
    Remove records whose tokenized text exceeds max_length tokens.

    Truncation is intentionally avoided: for NER, truncating silently drops
    entity labels at the end of the text. For QA and CLS, truncating cuts off
    the tail of a call transcript, which may contain the most relevant content.
    Filtering ensures every retained record fits within max_length without loss.

    Args:
        records:    list of dicts (raw data records)
        tokenizer:  HuggingFace tokenizer
        text_field: key in each record that holds the raw text
        max_length: maximum allowed token count (inclusive of special tokens)

    Returns:
        (filtered_records, stats) where stats is a dict:
            {
              "before":   int  — total records before filtering,
              "after":    int  — records kept,
              "removed":  int  — records dropped,
              "pct_kept": float — percentage retained,
            }
    """
    kept = []
    for record in records:
        text = str(record.get(text_field) or '')
        if count_tokens(text, tokenizer) <= max_length:
            kept.append(record)

    n_before = len(records)
    n_after = len(kept)
    stats = {
        'before':   n_before,
        'after':    n_after,
        'removed':  n_before - n_after,
        'pct_kept': round(100 * n_after / max(n_before, 1), 1),
    }
    return kept, stats


# ---------------------------------------------------------------------------
# Label map builder
# ---------------------------------------------------------------------------

def build_label_maps(
    ner_path: str,
    cls_path: str,
    qa_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> dict:
    """
    Build label maps from raw data files and optionally save to JSON.

    Returns a nested dict:
        {
          "ner":  { "label_to_id": {...}, "id_to_label": {...} },
          "cls":  { "main_cat2id": {...}, "id_to_main_cat": {...}, ... },
          "qa":   { "head_config": {...} },
        }
    """
    # NER labels are fixed
    ner_label_to_id = {label: i for i, label in enumerate(NER_LABELS)}
    ner_id_to_label = {str(i): label for label, i in ner_label_to_id.items()}

    # Classification labels derived from data
    cls_records = json.load(open(cls_path, encoding='utf-8'))
    df = pd.DataFrame(cls_records)
    df['main_category'] = df['category'].map(SUB_TO_MAIN).fillna('Unknown')
    df['priority'] = df['priority'].astype(str)

    # Filter out rows with missing values
    df = df[df['category'].notna() & df['intervention'].notna()]

    main_categories = sorted(df['main_category'].unique().tolist())
    sub_categories = sorted(df['category'].unique().tolist())
    interventions = sorted(df['intervention'].unique().tolist())
    priorities = sorted([p for p in df['priority'].unique().tolist() if p.strip()])

    main_cat2id = {c: i for i, c in enumerate(main_categories)}
    sub_cat2id = {c: i for i, c in enumerate(sub_categories)}
    interv2id = {c: i for i, c in enumerate(interventions)}
    priority2id = {c: i for i, c in enumerate(priorities)}

    label_maps = {
        "ner": {
            "label_to_id": ner_label_to_id,
            "id_to_label": ner_id_to_label,
        },
        "cls": {
            "main_cat2id": main_cat2id,
            "id_to_main_cat": {str(v): k for k, v in main_cat2id.items()},
            "sub_cat2id": sub_cat2id,
            "id_to_sub_cat": {str(v): k for k, v in sub_cat2id.items()},
            "interv2id": interv2id,
            "id_to_interv": {str(v): k for k, v in interv2id.items()},
            "priority2id": priority2id,
            "id_to_priority": {str(v): k for k, v in priority2id.items()},
        },
        "qa": {
            "head_config": QA_HEAD_CONFIG,
        },
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(label_maps, f, indent=2)
        print(f"Label maps saved to {output_path}")

    return label_maps


# ---------------------------------------------------------------------------
# Multi-Task Data Module
# ---------------------------------------------------------------------------

class MultiTaskDataModule:
    """
    Wraps all three task datasets and provides train/val DataLoaders.

    Usage:
        dm = MultiTaskDataModule(config, tokenizer, label_maps)
        dm.setup()
        train_loaders = dm.get_train_loaders(batch_size=16)
        val_loaders   = dm.get_val_loaders(batch_size=16)
    """

    def __init__(self, config: dict, tokenizer, label_maps: dict):
        self.config = config
        self.tokenizer = tokenizer
        self.label_maps = label_maps

        # Populated by setup()
        self.ner_train = self.ner_val = None
        self.cls_train = self.cls_val = None
        self.qa_train = self.qa_val = None

    def setup(self):
        data_cfg = self.config['data']
        test_size = data_cfg['test_size']
        seed = data_cfg['random_seed']
        max_len = self.config['tokenizer']['max_length']

        # --- NER  (FILTERED — token-level labels: truncation would corrupt them) ---
        with open(data_cfg['ner_path'], encoding='utf-8') as f:
            ner_records = [json.loads(line) for line in f if line.strip()]
        ner_records, ner_stats = filter_by_token_length(ner_records, self.tokenizer, 'text', max_len)
        print(f"NER  token filter : kept {ner_stats['after']}/{ner_stats['before']} "
              f"({ner_stats['pct_kept']}%) — {ner_stats['removed']} over-length removed")
        ner_train_r, ner_val_r = train_test_split(ner_records, test_size=test_size, random_state=seed)
        self.ner_train = NERDataset(ner_train_r, self.tokenizer, self.label_maps['ner']['label_to_id'], max_len)
        self.ner_val = NERDataset(ner_val_r, self.tokenizer, self.label_maps['ner']['label_to_id'], max_len)

        # --- Classification  (TRUNCATED — sentence-level labels, no corruption risk) ---
        with open(data_cfg['classification_path'], encoding='utf-8') as f:
            cls_records = json.load(f)
        known_subs = set(self.label_maps['cls']['sub_cat2id'].keys())
        known_priorities = set(self.label_maps['cls']['priority2id'].keys())
        cls_records = [
            r for r in cls_records
            if r.get('category') in known_subs
            and str(r.get('priority', '')).strip() in known_priorities
        ]
        cls_stats = {'before': len(cls_records), 'after': len(cls_records), 'removed': 0, 'pct_kept': 100.0}
        print(f"CLS  truncation   : {cls_stats['after']} records kept, long narratives truncated at {max_len}")
        cls_train_r, cls_val_r = train_test_split(cls_records, test_size=test_size, random_state=seed)
        self.cls_train = ClassificationDataset(cls_train_r, self.tokenizer, self.label_maps['cls'], max_len)
        self.cls_val = ClassificationDataset(cls_val_r, self.tokenizer, self.label_maps['cls'], max_len)

        # --- QA  (TRUNCATED — sentence-level labels, no corruption risk) ---
        with open(data_cfg['qa_path'], encoding='utf-8') as f:
            qa_records = json.load(f)
        qa_stats = {'before': len(qa_records), 'after': len(qa_records), 'removed': 0, 'pct_kept': 100.0}
        print(f"QA   truncation   : {qa_stats['after']} records kept, long transcripts truncated at {max_len}")
        qa_train_r, qa_val_r = train_test_split(qa_records, test_size=test_size, random_state=seed)
        self.qa_train = QADataset(qa_train_r, self.tokenizer, max_len)
        self.qa_val = QADataset(qa_val_r, self.tokenizer, max_len)

        # Store filter/truncation stats so the trainer can log them to MLflow
        self.filter_stats = {'ner': ner_stats, 'cls': cls_stats, 'qa': qa_stats}

        print(f"\nFinal dataset sizes:")
        print(f"  NER: {len(self.ner_train)} train / {len(self.ner_val)} val")
        print(f"  CLS: {len(self.cls_train)} train / {len(self.cls_val)} val")
        print(f"  QA:  {len(self.qa_train)} train / {len(self.qa_val)} val")

    def get_train_loaders(self, batch_size: int) -> Dict[str, DataLoader]:
        return {
            'ner': DataLoader(self.ner_train, batch_size=batch_size, shuffle=True, collate_fn=_ner_collate),
            'cls': DataLoader(self.cls_train, batch_size=batch_size, shuffle=True, collate_fn=_cls_collate),
            'qa':  DataLoader(self.qa_train,  batch_size=batch_size, shuffle=True, collate_fn=_qa_collate),
        }

    def get_val_loaders(self, batch_size: int) -> Dict[str, DataLoader]:
        return {
            'ner': DataLoader(self.ner_val, batch_size=batch_size, shuffle=False, collate_fn=_ner_collate),
            'cls': DataLoader(self.cls_val, batch_size=batch_size, shuffle=False, collate_fn=_cls_collate),
            'qa':  DataLoader(self.qa_val,  batch_size=batch_size, shuffle=False, collate_fn=_qa_collate),
        }
