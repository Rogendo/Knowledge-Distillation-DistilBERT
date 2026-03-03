import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from sklearn.metrics import confusion_matrix
import os
import sys

def load_data(file_path):
    """Loads the JSONL dataset."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def align_labels_with_tokens(tokenizer, text, entities):
    """Aligns character-level entity annotations with token-level IOB2 labels."""
    tokenized_inputs = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=512)
    offset_mapping = tokenized_inputs["offset_mapping"]

    labels = ['O'] * len(offset_mapping)
    
    for entity in entities:
        start_char, end_char, label = entity['start'], entity['end'], entity['label']
        
        entity_tokens = []
        for i, (start, end) in enumerate(offset_mapping):
            if start is None or end is None:
                continue
            if max(start, start_char) < min(end, end_char):
                entity_tokens.append(i)
        
        if entity_tokens:
            labels[entity_tokens[0]] = f"B-{label}"
            for token_idx in entity_tokens[1:]:
                labels[token_idx] = f"I-{label}"
                
    return labels

def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    """Plots and saves a token-level confusion matrix."""
    true_flat = [label for seq in y_true for label in seq]
    pred_flat = [label for seq in y_pred for label in seq]
    all_labels = sorted(list(set(true_flat) | set(pred_flat)))
    
    # Remove 'O' from matrix for better readability if it exists
    if 'O' in all_labels:
        all_labels.remove('O')

    cm = confusion_matrix(true_flat, pred_flat, labels=all_labels)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=all_labels, yticklabels=all_labels,
                cbar_kws={'shrink': 0.8})
    plt.title('Token-level Confusion Matrix for NER', fontsize=18, pad=20)
    plt.ylabel('True Label', fontsize=15)
    plt.xlabel('Predicted Label', fontsize=15)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {filename}")

def main():
    """Main function to run the NER model evaluation."""
    model_path = "./ner-distilbert-en-synthetic-v2"
    data_path = "./ner_synthetic_dataset_v2.jsonl"
    label_map_path = os.path.join(model_path, "label_mappings.json")
    output_dir = "ner_evaluation_results"
    os.makedirs(output_dir, exist_ok=True)

    # Load external label mapping
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    id_to_label = {int(k): v for k, v in label_map['id_to_label'].items()}

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_data = load_data(data_path)

    y_true = []
    y_pred = []

    for example in test_data:
        text = example['text']
        entities = example['entities']
        
        true_labels_tokens = align_labels_with_tokens(tokenizer, text, entities)
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        
        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
        word_ids = inputs.word_ids()
        
        pred_labels_for_seq = []
        true_labels_for_seq = []
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == previous_word_idx:
                continue
            
            true_labels_for_seq.append(true_labels_tokens[i])

            label_id = predictions[i]
            # Use the correct, externally loaded label map
            label = id_to_label.get(label_id, 'O')
            
            if label != 'O':
                # Check if the previous token belongs to the same word
                is_continuation = (i > 0 and word_ids[i-1] == word_idx)
                if is_continuation:
                    pred_labels_for_seq.append(f"I-{label}")
                else:
                    pred_labels_for_seq.append(f"B-{label}")
            else:
                pred_labels_for_seq.append('O')

            previous_word_idx = word_idx
        
        # Ensure sequences are of the same length
        min_len = min(len(true_labels_for_seq), len(pred_labels_for_seq))
        y_true.append(true_labels_for_seq[:min_len])
        y_pred.append(pred_labels_for_seq[:min_len])

    report = classification_report(y_true, y_pred, scheme=IOB2, mode='strict', output_dict=True, zero_division=0)
    
    print("NER Classification Report (Entity-level):")
    print(f"{ '':<25} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    
    entity_labels = sorted([key for key in report.keys() if key not in ['micro avg', 'macro avg', 'weighted avg']])

    for label in entity_labels:
        metrics = report[label]
        print(f"{label:<25} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} {metrics['f1-score']:>10.3f} {metrics['support']:>10}")

    if 'micro avg' in report:
        print(f"\nOverall Accuracy (Micro F1): {report['micro avg']['f1-score']:.3f}")

    plot_confusion_matrix(y_true, y_pred, filename=os.path.join(output_dir, "ner_confusion_matrix.png"))

if __name__ == "__main__":
    main()