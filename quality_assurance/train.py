import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup
from typing import List, Dict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
import os
import json
import yaml
import logging
from tqdm import tqdm

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
print("QA Heads Configuration:", qa_heads_config)

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
def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    all_preds, all_labels = {h: [] for h in qa_heads_config}, {h: [] for h in qa_heads_config}
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            for head in qa_heads_config:
                all_preds[head].extend((outputs["logits"][head].cpu().numpy() > threshold).astype(int))
                all_labels[head].extend(batch['labels'][head].cpu().numpy().astype(int))
    metrics = {}
    for head in qa_heads_config:
        y_true, y_pred = np.array(all_labels[head]), np.array(all_preds[head])
        metrics[f"{head}_accuracy"] = accuracy_score(y_true, y_pred)
        metrics[f"{head}_precision"] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics[f"{head}_recall"] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics[f"{head}_f1_score"] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    return metrics

# --- Training Function ---
def train_model(config):
    # Setup Logging
    logging.basicConfig(level=config['logging']['level'], format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(config['logging']['log_file']), logging.StreamHandler()] if config['logging']['console_output'] else [logging.FileHandler(config['logging']['log_file'])])

    # MLflow Setup
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # Data Loading and Splitting
    df = pd.read_json(config['data']['train_data_path'])
    train_df, test_df = train_test_split(df, test_size=config['data']['test_size'], random_state=config['data']['random_seed'])
    train_df, val_df = train_test_split(train_df, test_size=config['data']['val_size'], random_state=config['data']['random_seed'])
    

    print(f"Training examples: {len(train_df)}")
    print(f"Validation examples: {len(val_df)}")
    print(f"Test examples: {len(test_df)}")

    # Save test split
    test_df.to_json(config['data']['test_data_path'], orient='records')
    logging.info(f"Test data saved to {config['data']['test_data_path']}")

    logging.info(f"Training examples: {len(train_df)}, Validation examples: {len(val_df)}, Test examples: {len(test_df)}")

    # Tokenizer and Datasets
    tokenizer = DistilBertTokenizer.from_pretrained(config['model']['base_model'])
    train_dataset = MultiHeadQADataset(train_df, tokenizer, config['tokenizer']['max_length'])
    val_dataset = MultiHeadQADataset(val_df, tokenizer, config['tokenizer']['max_length'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], collate_fn=collate_fn, shuffle=config['data']['shuffle'])
    val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], collate_fn=collate_fn)

    # Model, Optimizer, Scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadQAClassifier(config['model']['base_model'], qa_heads_config, config['model']['dropout']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['training']['warmup_steps'], num_training_steps=len(train_dataloader) * config['training']['num_epochs'])

    # Training Loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    with mlflow.start_run(run_name="Config-driven QA Model Training") as run:
        mlflow.log_params({**config['model'], **config['data'], **config['tokenizer'], **config['training']})
        for epoch in range(config['training']['num_epochs']):
            model.train()
            total_loss = 0
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()
                outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels={k: v.to(device) for k, v in batch['labels'].items()})
                loss = outputs['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_dataloader)
            val_metrics = evaluate_model(model, val_dataloader, device)
            avg_val_loss = sum(val_metrics.values())/len(val_metrics) # A simple proxy for validation loss
            
            logging.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            mlflow.log_metrics({"train_loss": avg_train_loss, "val_loss": avg_val_loss, **val_metrics}, step=epoch)

            # Early Stopping and Model Saving
            if avg_val_loss < best_val_loss - config['training']['early_stopping_min_delta']:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                if config['output']['save_best_model']:
                    output_dir = os.path.join(config['model']['output_dir'], f"qa_model_best_epoch{epoch+1}")
                    os.makedirs(output_dir, exist_ok=True)
                    model.bert.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                    with open(config['output']['metrics_file_path'], 'w') as f: json.dump(val_metrics, f)
                    logging.info(f"Saved best model to {output_dir}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config['training']['early_stopping_patience']:
                    logging.info("Early stopping triggered.")
                    break
        
        # Final Model Save
        if config.get('output', {}).get('save_final_model', True):
            final_output_dir = os.path.join(config['model']['output_dir'], "qa_model_final")
            os.makedirs(final_output_dir, exist_ok=True)
            model.bert.save_pretrained(final_output_dir)
            tokenizer.save_pretrained(final_output_dir)
            torch.save(model.state_dict(), os.path.join(final_output_dir, "pytorch_model.bin"))
            mlflow.pytorch.log_model(model, "model", registered_model_name=config['mlflow']['registered_model_name'])

if __name__ == "__main__":
    config = load_config()
    train_model(config)
