"""
Inference module for the Attention Fusion Multi-Task Model.

Loads a saved checkpoint (model weights + label maps + tokenizer) and
provides three task-specific prediction methods:
  - predict_ner(texts)            → list of [(token, label), ...]
  - predict_classification(texts) → list of dicts with 4 label predictions
  - predict_qa(texts)             → list of dicts with per-head binary scores
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from model import AttentionFusionModel
from datasets import QA_HEAD_CONFIG


class AttentionFusionInference:
    """
    Loads the trained model from a checkpoint directory and provides
    task-specific prediction methods.

    The checkpoint directory must contain:
        model.pt          - model state dict
        label_maps.json   - label maps produced by build_label_maps()
        tokenizer files   - saved by tokenizer.save_pretrained()

    Example:
        inf = AttentionFusionInference("./attention_fusion_model")
        ner_out = inf.predict_ner(["Police arrested John in Nairobi."])
        cls_out = inf.predict_classification(["I need help with my daughter."])
        qa_out  = inf.predict_qa(["Hello, child helpline, how can I help?"])
    """

    def __init__(
        self,
        model_dir: str,
        device: str = None,
        backbone_name: str = None,
        dropout: float = 0.0,
        qa_threshold: float = 0.5,
    ):
        self.model_dir = model_dir
        self.qa_threshold = qa_threshold
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # Load label maps
        maps_path = os.path.join(model_dir, 'label_maps.json')
        if not os.path.exists(maps_path):
            raise FileNotFoundError(f"label_maps.json not found in {model_dir}")
        with open(maps_path, encoding='utf-8') as f:
            self.label_maps = json.load(f)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Determine backbone name
        if backbone_name is None:
            # Try to read from a saved config in the dir, else fall back to default
            cfg_path = os.path.join(model_dir, 'config.json')
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    saved_cfg = json.load(f)
                backbone_name = saved_cfg.get('_name_or_path', 'distilbert-base-cased')
            else:
                backbone_name = 'distilbert-base-cased'

        cls_maps = self.label_maps['cls']
        num_main = len(cls_maps['main_cat2id'])
        num_sub = len(cls_maps['sub_cat2id'])
        num_interv = len(cls_maps['interv2id'])
        num_priority = len(cls_maps['priority2id'])

        # Build and load model
        self.model = AttentionFusionModel(
            backbone_name=backbone_name,
            dropout=dropout,
            num_ner_labels=len(self.label_maps['ner']['label_to_id']),
            num_main_cat=num_main,
            num_sub_cat=num_sub,
            num_intervention=num_interv,
            num_priority=num_priority,
            qa_heads_config=self.label_maps['qa']['head_config'],
        )

        weights_path = os.path.join(model_dir, 'model.pt')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"model.pt not found in {model_dir}")
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Build reverse label maps (int key → string)
        ner_maps = self.label_maps['ner']
        self._ner_id_to_label = {int(k): v for k, v in ner_maps['id_to_label'].items()}
        self._cls_id_to_main = {int(k): v for k, v in cls_maps['id_to_main_cat'].items()}
        self._cls_id_to_sub = {int(k): v for k, v in cls_maps['id_to_sub_cat'].items()}
        self._cls_id_to_interv = {int(k): v for k, v in cls_maps['id_to_interv'].items()}
        self._cls_id_to_priority = {int(k): v for k, v in cls_maps['id_to_priority'].items()}

        print(f"Model loaded from {model_dir} on {self.device}")

    # ------------------------------------------------------------------
    # NER
    # ------------------------------------------------------------------

    def predict_ner(self, texts: List[str]) -> List[List[Tuple[str, str]]]:
        """
        Predict named entities for a list of texts.

        Returns:
            List of [(token_str, label_str), ...] per text.
            Special tokens ([CLS], [SEP], [PAD]) are excluded.
        """
        results = []
        for text in texts:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=False,
                return_offsets_mapping=True,
                return_tensors='pt',
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            offset_mapping = encoding['offset_mapping'].squeeze(0).tolist()

            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task='ner',
                )
            preds = out['logits'].argmax(dim=-1).squeeze(0).cpu().tolist()

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

            sentence_result = []
            for token, pred_id, (tok_start, tok_end) in zip(tokens, preds, offset_mapping):
                # Skip special tokens (offset (0,0)) and padding
                if tok_start == 0 and tok_end == 0:
                    continue
                label = self._ner_id_to_label.get(pred_id, 'O')
                sentence_result.append((token, label))

            results.append(sentence_result)
        return results

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def predict_classification(self, texts: List[str]) -> List[Dict[str, str]]:
        """
        Predict case classification for a list of texts.

        Returns:
            List of dicts:
            {
              "main_category": str,
              "sub_category":  str,
              "intervention":  str,
              "priority":      str,
            }
        """
        results = []
        for text in texts:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=False,
                return_tensors='pt',
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task='cls',
                )
            logits = out['logits']

            main_id = logits['main'].argmax(dim=-1).item()
            sub_id = logits['sub'].argmax(dim=-1).item()
            interv_id = logits['intervention'].argmax(dim=-1).item()
            priority_id = logits['priority'].argmax(dim=-1).item()

            results.append({
                'main_category': self._cls_id_to_main.get(main_id, 'Unknown'),
                'sub_category':  self._cls_id_to_sub.get(sub_id, 'Unknown'),
                'intervention':  self._cls_id_to_interv.get(interv_id, 'Unknown'),
                'priority':      self._cls_id_to_priority.get(priority_id, 'Unknown'),
            })
        return results

    # ------------------------------------------------------------------
    # QA
    # ------------------------------------------------------------------

    def predict_qa(self, texts: List[str]) -> List[Dict[str, List[int]]]:
        """
        Predict QA quality scores for a list of texts.

        Returns:
            List of dicts with per-head binary predictions:
            {
              "opening":       [0|1],
              "listening":     [0|1, 0|1, 0|1, 0|1, 0|1],
              "proactiveness": [0|1, 0|1, 0|1],
              "resolution":    [0|1, 0|1, 0|1, 0|1, 0|1],
              "hold":          [0|1, 0|1],
              "closing":       [0|1],
            }
        """
        results = []
        for text in texts:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=False,
                return_tensors='pt',
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task='qa',
                )
            logits = out['logits']

            head_preds = {}
            for head in QA_HEAD_CONFIG:
                probs = logits[head].sigmoid().squeeze(0).cpu().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                head_preds[head] = [int(p > self.qa_threshold) for p in probs]

            results.append(head_preds)
        return results


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run inference with the Attention Fusion model')
    parser.add_argument('--model_dir', default='./attention_fusion_model', help='Checkpoint directory')
    args = parser.parse_args()

    inf = AttentionFusionInference(args.model_dir)

    sample_texts = [
        "Hello, I'm calling from Nairobi. My daughter Sarah, aged 12, was assaulted by her teacher.",
        "Hello, child helpline, how can I help you today? I need help urgently for my son.",
    ]

    print("\n--- NER ---")
    for text, result in zip(sample_texts, inf.predict_ner(sample_texts)):
        print(f"Text: {text[:60]}...")
        entities = [(tok, lbl) for tok, lbl in result if lbl != 'O']
        print(f"Entities: {entities}\n")

    print("\n--- Classification ---")
    for text, result in zip(sample_texts, inf.predict_classification(sample_texts)):
        print(f"Text: {text[:60]}...")
        print(f"Result: {result}\n")

    print("\n--- QA ---")
    for text, result in zip(sample_texts, inf.predict_qa(sample_texts)):
        print(f"Text: {text[:60]}...")
        print(f"Result: {result}\n")
