
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from typing import List, Dict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import json
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration Loading ---
def load_config(config_path="config.yaml"):
    """Loads YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Heads and Labels ---
HEAD_SUBMETRIC_LABELS = {
    "opening": ["Use of call opening phrase"],
    "listening": ["Caller was not interrupted", "Empathizes with the caller", "Paraphrases or rephrases the issue", "Uses 'please' and 'thank you'", "Does not hesitate or sound unsure"],
    "proactiveness": ["Willing to solve extra issues", "Confirms satisfaction with action points", "Follows up on case updates"],
    "resolution": ["Gives accurate information", "Correct language use", "Consults if unsure", "Follows correct steps", "Explains solution process clearly"],
    "hold": ["Explains before placing on hold", "Thanks caller for holding"],
    "closing": ["Proper call closing phrase used"]
}

qa_heads_config = {head: len(labels) for head, labels in HEAD_SUBMETRIC_LABELS.items()}

# --- Dataset Class ---
class MultiHeadQADataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        encoding = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        labels = {head: torch.tensor(item['labels'][head], dtype=torch.float) for head in qa_heads_config}
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }

# --- Collate Function ---
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = {key: torch.stack([item['labels'][key] for item in batch]) for key in batch[0]['labels']}
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# --- Model Class ---
class MultiHeadQAClassifier(nn.Module):
    def __init__(self, model_name, heads_config, dropout):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({head: nn.Linear(hidden_size, output_dim) for head, output_dim in heads_config.items()})

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(output.last_hidden_state[:, 0])
        logits, losses, loss_total = {}, {}, 0
        for head_name, head_layer in self.heads.items():
            out = head_layer(pooled_output)
            logits[head_name] = torch.sigmoid(out)
            if labels is not None:
                loss = nn.BCEWithLogitsLoss()(out, labels[head_name])
                losses[head_name], loss_total = loss.item(), loss_total + loss
        return {"logits": logits, "loss": loss_total if labels is not None else None, "losses": losses if labels is not None else None}

# --- Evaluation Function ---
def evaluate_model(model, dataloader, device, output_dir, threshold=0.5):
    model.eval()
    all_preds, all_labels = {h: [] for h in qa_heads_config}, {h: [] for h in qa_heads_config}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            for head in qa_heads_config:
                all_preds[head].extend((outputs["logits"][head].cpu().numpy() > threshold).astype(int))
                all_labels[head].extend(batch['labels'][head].cpu().numpy().astype(int))
    
    metrics = {}
    for head in qa_heads_config:
        y_true = np.array(all_labels[head])
        y_pred = np.array(all_preds[head])
        
        if y_true.shape != y_pred.shape:
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]

        cm = multilabel_confusion_matrix(y_true, y_pred, labels=range(y_pred.shape[1]))
        
        metrics[head] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='micro', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='micro', zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average='micro', zero_division=0),
        }
        
        # Plot and save confusion matrices
        for i, label_cm in enumerate(cm):
            plt.figure(figsize=(6, 5))
            sns.heatmap(label_cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Predicted Negative', 'Predicted Positive'],
                        yticklabels=['Actual Negative', 'Actual Positive'])
            plt.title(f'Confusion Matrix for {head} - {HEAD_SUBMETRIC_LABELS[head][i]}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            # Sanitize label name for filename
            label_filename = HEAD_SUBMETRIC_LABELS[head][i].replace(' ', '_').lower()
            filename = os.path.join(output_dir, f'cm_{head}_{label_filename}.png')
            plt.savefig(filename)
            plt.close()

    return metrics

def main():
    config = load_config()
    
    # Create directory for confusion matrices
    cm_output_dir = "confusion_matrices"
    os.makedirs(cm_output_dir, exist_ok=True)

    # --- Data Loading ---
    test_df = pd.read_json(config['data']['test_data_path'], orient='records')
    
    print(f"Test examples: {len(test_df)}")

    # --- Tokenizer and Dataset ---
    model_path = os.path.join(config['model']['output_dir'], "qa_model_final")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    test_dataset = MultiHeadQADataset(test_df, tokenizer, config['tokenizer']['max_length'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], collate_fn=collate_fn)

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadQAClassifier(model_path, qa_heads_config, config['model']['dropout'])
    model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device))
    model.to(device)

    # --- Evaluation ---
    evaluation_metrics = evaluate_model(model, test_dataloader, device, cm_output_dir)

    # --- Save Metrics ---
    output_file = "evaluation_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)
    
    print(f"Evaluation metrics saved to {output_file}")
    print(f"Confusion matrices saved in '{cm_output_dir}' directory.")

if __name__ == "__main__":
    main()
