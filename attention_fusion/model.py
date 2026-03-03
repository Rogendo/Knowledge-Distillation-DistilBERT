"""
Attention Fusion Multi-Task NLP Model

Single DistilBERT backbone shared across three tasks:
  - NER (token-level, 10 labels)
  - Case Classification (sentence-level, 4 heads)
  - Quality Assurance (sentence-level, 6 binary heads)

Each sentence-level task gets its own TaskAttentionPooling module that learns
which tokens matter for that specific task.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel
from typing import Dict, Optional


class TaskAttentionPooling(nn.Module):
    """
    Learned attention pooling for sentence-level tasks.

    Each task has its own instance so it can learn task-specific token importance.
    Padding tokens are masked with -inf before softmax so they contribute no weight.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len]  (1=real token, 0=padding)
        Returns:
            pooled: [batch, hidden_dim]
        """
        scores = self.attention(hidden_states)          # [batch, seq_len, 1]
        padding_mask = (attention_mask == 0).unsqueeze(-1)  # [batch, seq_len, 1]
        scores = scores.masked_fill(padding_mask, float('-inf'))
        weights = torch.softmax(scores, dim=1)          # [batch, seq_len, 1]
        pooled = (weights * hidden_states).sum(dim=1)   # [batch, hidden_dim]
        return pooled


class AttentionFusionModel(nn.Module):
    """
    Single-backbone multi-task model with per-task attention fusion.

    Architecture:
        Input → DistilBERT backbone → hidden states [batch, seq_len, 768]
            ├── NER:  linear transform → token classifier (10 labels)
            ├── CLS:  cls_attention pooling → 4 classification heads
            └── QA:   qa_attention pooling  → 6 binary heads
    """

    # Default QA head configuration (head_name → num_binary_outputs)
    DEFAULT_QA_HEADS = {
        "opening": 1,
        "listening": 5,
        "proactiveness": 3,
        "resolution": 5,
        "hold": 2,
        "closing": 1,
    }

    def __init__(
        self,
        backbone_name: str = "distilbert-base-cased",
        dropout: float = 0.1,
        num_ner_labels: int = 10,
        num_main_cat: int = 8,
        num_sub_cat: int = 77,
        num_intervention: int = 16,
        num_priority: int = 3,
        qa_heads_config: Optional[Dict[str, int]] = None,
    ):
        super().__init__()

        if qa_heads_config is None:
            qa_heads_config = self.DEFAULT_QA_HEADS

        self.backbone = DistilBertModel.from_pretrained(backbone_name)
        hidden_dim = self.backbone.config.dim  # 768 for distilbert-base

        self.dropout = nn.Dropout(dropout)

        # --- NER head (token-level) ---
        # Linear transform preserving sequence dimension, then per-token classifier
        self.ner_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ner_classifier = nn.Linear(hidden_dim, num_ner_labels)

        # --- Classification head (sentence-level) ---
        self.cls_attention = TaskAttentionPooling(hidden_dim)
        self.cls_main = nn.Linear(hidden_dim, num_main_cat)
        self.cls_sub = nn.Linear(hidden_dim, num_sub_cat)
        self.cls_intervention = nn.Linear(hidden_dim, num_intervention)
        self.cls_priority = nn.Linear(hidden_dim, num_priority)

        # --- QA head (sentence-level) ---
        self.qa_attention = TaskAttentionPooling(hidden_dim)
        self.qa_heads = nn.ModuleDict({
            head: nn.Linear(hidden_dim, size)
            for head, size in qa_heads_config.items()
        })

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: str,
        labels=None,
    ) -> dict:
        """
        Shared forward pass routed by task name.

        Args:
            input_ids:      [batch, seq_len]
            attention_mask: [batch, seq_len]
            task:           one of "ner", "cls", "qa"
            labels:         task-specific labels dict (optional, for training)
        Returns:
            dict with keys "logits", "loss" (if labels provided), "task"
        """
        backbone_out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = backbone_out.last_hidden_state  # [batch, seq_len, 768]

        if task == "ner":
            return self._forward_ner(hidden_states, labels)
        elif task == "cls":
            return self._forward_cls(hidden_states, attention_mask, labels)
        elif task == "qa":
            return self._forward_qa(hidden_states, attention_mask, labels)
        else:
            raise ValueError(f"Unknown task '{task}'. Choose from: ner, cls, qa")

    def _forward_ner(self, hidden_states: torch.Tensor, labels=None) -> dict:
        x = self.ner_transform(hidden_states)   # [batch, seq_len, 768]
        logits = self.ner_classifier(x)          # [batch, seq_len, num_ner_labels]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        return {"logits": logits, "loss": loss, "task": "ner"}

    def _forward_cls(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        labels=None,
    ) -> dict:
        pooled = self.cls_attention(hidden_states, attention_mask)  # [batch, 768]
        pooled = self.dropout(pooled)

        logits = {
            "main": self.cls_main(pooled),
            "sub": self.cls_sub(pooled),
            "intervention": self.cls_intervention(pooled),
            "priority": self.cls_priority(pooled),
        }

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = (
                loss_fct(logits["main"], labels["main_category_id"])
                + loss_fct(logits["sub"], labels["sub_category_id"])
                + loss_fct(logits["intervention"], labels["intervention_id"])
                + loss_fct(logits["priority"], labels["priority_id"])
            )

        return {"logits": logits, "loss": loss, "task": "cls"}

    def _forward_qa(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        labels=None,
    ) -> dict:
        pooled = self.qa_attention(hidden_states, attention_mask)  # [batch, 768]
        pooled = self.dropout(pooled)

        logits = {
            head: layer(pooled)
            for head, layer in self.qa_heads.items()
        }

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = sum(
                loss_fct(logits[head], labels[head].float())
                for head in self.qa_heads
            )

        return {"logits": logits, "loss": loss, "task": "qa"}
