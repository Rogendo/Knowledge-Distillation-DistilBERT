
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import os
import yaml

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


def run_inference(texts, model, tokenizer, device, threshold=0.5):
    model.eval()
    for text in texts:
        print(f"--- Analyzing Text: \"{text}\" ---")
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
        
        for head, labels in HEAD_SUBMETRIC_LABELS.items():
            print(f"\n  Head: {head}")
            predictions = (outputs["logits"][head].cpu().numpy() > threshold).astype(int).flatten()
            for label, prediction in zip(labels, predictions):
                print(f"    - {label}: {'Yes' if prediction == 1 else 'No'}")

def main():
    config = load_config()
    
    # --- Load Model and Tokenizer ---
    model_path = os.path.join(config['model']['output_dir'], "qa_model_final")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadQAClassifier(model_path, qa_heads_config, config['model']['dropout'])
    model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device))
    model.to(device)

    # --- Example Texts for Inference ---
    example_texts = [
        "Hello, thank you for calling. How can I help you today?",
        "Yeah, what do you want?",
        "I understand your frustration, and I'm here to help you resolve this issue.",
        "I will put you on hold for a moment."
    ]

    run_inference(example_texts, model, tokenizer, device)

if __name__ == "__main__":
    main()
