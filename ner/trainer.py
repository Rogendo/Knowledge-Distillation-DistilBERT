
import pandas as pd
import yaml
import json
import os
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    Trainer, 
    TrainingArguments,
    set_seed
)
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def setup_data(config: dict) -> tuple:
    """Load and prepare dataset."""
    file_path = config['data']['dataset_path']
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    print(f"Loading dataset from {file_path}")
    df = pd.read_json(file_path, lines=True)
    dataset = Dataset.from_pandas(df)
    
    # Create label mappings
    unique_labels = ['O'] + [ent['label'] for record in df['entities'] for ent in record]
    label_to_id = {label: idx for idx, label in enumerate(set(unique_labels))}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    print(f"Found {len(label_to_id)} unique labels: {list(label_to_id.keys())}")
    
    return dataset, df, label_to_id, id_to_label


def setup_tokenizer_and_model(config: dict, num_labels: int):
    """Initialize tokenizer and model."""
    model_name = config['model']['model_name']
    tokenizer_name = config['model'].get('tokenizer_name', model_name)
    
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    print(f"Loading model: {model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    return tokenizer, model


def create_tokenize_function(tokenizer, label_to_id: dict, config: dict):
    """Create tokenization function with configuration."""
    tokenizer_config = config['tokenizer']
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples['text'],
            truncation=tokenizer_config['truncation'],
            is_split_into_words=False,
            padding=tokenizer_config['padding'],
            max_length=tokenizer_config['max_length'],
            return_offsets_mapping=tokenizer_config['return_offsets_mapping'],
        )

        labels = []
        for batch_index in range(len(examples['text'])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            example_labels = [label_to_id['O']] * len(word_ids)
            offset_mapping = tokenized_inputs['offset_mapping'][batch_index]

            for entity in examples['entities'][batch_index]:
                start = entity['start']
                end = entity['end']
                label = label_to_id[entity['label']]

                for idx, word_id in enumerate(word_ids):
                    if word_id is None:
                        continue
                    token_start, token_end = offset_mapping[idx]
                    if token_start >= start and token_end <= end:
                        example_labels[idx] = label
        
            labels.append(example_labels)

        tokenized_inputs['labels'] = labels
        tokenized_inputs.pop('offset_mapping')
        return tokenized_inputs
    
    return tokenize_and_align_labels


def create_compute_metrics_function(id_to_label: dict, config: dict):
    """Create compute metrics function."""
    average_method = config['compute_metrics']['average_method']
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        flat_predictions = [item for sublist in true_predictions for item in sublist]
        flat_labels = [item for sublist in true_labels for item in sublist]

        accuracy = accuracy_score(flat_labels, flat_predictions)
        f1 = f1_score(flat_labels, flat_predictions, average=average_method, zero_division=0)
        recall = recall_score(flat_labels, flat_predictions, average=average_method, zero_division=0)
        precision = precision_score(flat_labels, flat_predictions, average=average_method, zero_division=0)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'recall': recall,
            'precision': precision
        }
    
    return compute_metrics


def setup_training_arguments(config: dict) -> TrainingArguments:
    """Setup training arguments from config."""
    training_config = config['training']
    
    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        warmup_ratio=training_config['warmup_ratio'],
        logging_dir=training_config['logging_dir'],
        logging_steps=training_config['logging_steps'],
        eval_strategy=training_config['eval_strategy'],
        eval_steps=training_config['eval_steps'],
        save_strategy=training_config['save_strategy'],
        save_steps=training_config['save_steps'],
        save_total_limit=training_config['save_total_limit'],
        load_best_model_at_end=training_config['load_best_model_at_end'],
        metric_for_best_model=training_config['metric_for_best_model'],
        greater_is_better=training_config['greater_is_better'],
        report_to=training_config['report_to']
    )
    
    return training_args


def create_ner_inference_function(model_path: str, tokenizer_path: str, id_to_label: dict):
    """Create NER inference function."""
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    def ner_detection_batch(sentences):
        """
        Detects named entities in a batch of sentences using the finetuned NER model.

        Args:
        sentences (list): A list of input sentences.

        Returns:
        list: A list of results for each sentence. Each result is a list of tuples 
              containing tokens of the classified entities
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        results = []
        for text in sentences:
            tokens = tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                is_split_into_words=False
            ).to(device)

            with torch.no_grad():
                output = model(**tokens)

            predictions = np.argmax(output.logits.detach().cpu().numpy(), axis=2)
            token_list = tokenizer.convert_ids_to_tokens(tokens['input_ids'].squeeze().tolist())
            labels = [id_to_label[label] for label in predictions[0]]

            sentence_results = []
            for token, label in zip(token_list, labels):
                if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                    sentence_results.append((token, label))

            results.append(sentence_results)

        return results
    
    return ner_detection_batch


def main(config_path: str):
    """Main training function."""
    # Load configuration
    config = load_config(config_path)
    
    # Set seed for reproducibility
    if 'seed' in config:
        set_seed(config['seed'])
    
    # Setup data
    dataset, df, label_to_id, id_to_label = setup_data(config)
    
    # Setup tokenizer and model
    tokenizer, model = setup_tokenizer_and_model(config, len(label_to_id))
    
    # Create tokenization function
    tokenize_and_align_labels = create_tokenize_function(tokenizer, label_to_id, config)
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
    
    # Split dataset
    test_size = config['data']['test_size']
    train_dataset = tokenized_dataset.train_test_split(test_size=test_size)['train']
    val_dataset = tokenized_dataset.train_test_split(test_size=test_size)['test']
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Setup training arguments
    training_args = setup_training_arguments(config)
    
    # Create compute metrics function
    compute_metrics = create_compute_metrics_function(id_to_label, config)
    
    # Initialize trainer (handle deprecation warning for newer transformers versions)
    try:
        # Try the new parameter name first (transformers >= 5.0)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
        )
    except TypeError:
        # Fall back to the old parameter name
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Evaluate
    print("Evaluating model...")
    metrics = trainer.evaluate()
    
    # Save metrics
    metrics_file = config['output']['metrics_file']
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")
    
    # Save model and tokenizer
    output_dir = config['output']['model_save_dir']
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label mappings
    with open(os.path.join(output_dir, "label_mappings.json"), "w") as f:
        json.dump({
            "label_to_id": label_to_id,
            "id_to_label": id_to_label
        }, f, indent=2)
    
    print(f"Model and tokenizer saved to {output_dir}")
    
    # Test inference function
    print("\nTesting inference...")
    ner_function = create_ner_inference_function(output_dir, output_dir, id_to_label)
    
    # Test with a sample
    example_sentences = [
        "Hello, I'm Vincent from Dar es Salaam. I need help with my 10-year-old daughter.",
        "Mwangi Kennedy was seriously bruised and was taken to the Hospital"
    ]
    
    detected_entities = ner_function(example_sentences)
    for i, entities in enumerate(detected_entities):
        print(f"Sample {i+1} entities: {entities[:5]}...")  # Show first 5 entities
    
    print("Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NER model with YAML configuration")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config_v2.yaml",
        help="Path to configuration YAML file"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        print("Please create a config.yaml file or specify a valid config path.")
        exit(1)
    
    main(args.config)
