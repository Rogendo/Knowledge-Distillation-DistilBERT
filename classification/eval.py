import os
import torch
import pandas as pd
import numpy as np
import json
import yaml
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_test_data(csv_path):
    """Load test data from CSV"""
    logger.info(f"Loading test data from: {csv_path}")
    test_df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(test_df)} test samples")
    return test_df

def load_model_and_tokenizer(model_path):
    """Load saved model and tokenizer"""
    from trainer import MultiTaskDistilBert  # Import your model class
    
    logger.info(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # You'll need to pass the correct number of classes
    # These should be loaded from your label files
    model = MultiTaskDistilBert.from_pretrained(model_path)
    model.eval()
    
    return model, tokenizer

def predict_batch(model, tokenizer, texts, device, max_length=512):
    """Make predictions on a batch of texts"""
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    encodings = {k: v.to(device) for k, v in encodings.items()}
    
    with torch.no_grad():
        outputs = model(**encodings)
    
    # Get predictions for all tasks
    preds_main = torch.argmax(outputs[0], dim=1).cpu().numpy()
    preds_sub = torch.argmax(outputs[1], dim=1).cpu().numpy()
    preds_interv = torch.argmax(outputs[2], dim=1).cpu().numpy()
    preds_priority = torch.argmax(outputs[3], dim=1).cpu().numpy()
    
    return preds_main, preds_sub, preds_interv, preds_priority

def evaluate_model(test_df, model, tokenizer, device, batch_size=16):
    """Evaluate model on test data"""
    logger.info("Starting evaluation...")
    
    all_preds_main = []
    all_preds_sub = []
    all_preds_interv = []
    all_preds_priority = []
    
    # Process in batches
    for i in range(0, len(test_df), batch_size):
        batch_texts = test_df['text'].iloc[i:i+batch_size].tolist()
        
        preds_main, preds_sub, preds_interv, preds_priority = predict_batch(
            model, tokenizer, batch_texts, device
        )
        
        all_preds_main.extend(preds_main)
        all_preds_sub.extend(preds_sub)
        all_preds_interv.extend(preds_interv)
        all_preds_priority.extend(preds_priority)
    
    # Get true labels
    labels_main = test_df['main_category_id'].values
    labels_sub = test_df['sub_category_id'].values
    labels_interv = test_df['intervention_id'].values
    labels_priority = test_df['priority_id'].values
    
    # Compute metrics
    metrics = {}
    
    def compute_task_metrics(task_name, labels, preds):
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )
        
        return {
            f"{task_name}_accuracy": accuracy,
            f"{task_name}_precision": precision,
            f"{task_name}_recall": recall,
            f"{task_name}_f1": f1
        }
    
    metrics.update(compute_task_metrics("main", labels_main, all_preds_main))
    metrics.update(compute_task_metrics("sub", labels_sub, all_preds_sub))
    metrics.update(compute_task_metrics("interv", labels_interv, all_preds_interv))
    metrics.update(compute_task_metrics("priority", labels_priority, all_preds_priority))
    
    # Average metrics
    metrics["avg_accuracy"] = np.mean([
        metrics["main_accuracy"],
        metrics["sub_accuracy"],
        metrics["interv_accuracy"],
        metrics["priority_accuracy"]
    ])
    
    metrics["avg_f1"] = np.mean([
        metrics["main_f1"],
        metrics["sub_f1"],
        metrics["interv_f1"],
        metrics["priority_f1"]
    ])
    
    return metrics, all_preds_main, all_preds_sub, all_preds_interv, all_preds_priority

def save_predictions(test_df, preds_main, preds_sub, preds_interv, preds_priority, output_path):
    """Save predictions to CSV"""
    results_df = test_df.copy()
    results_df['pred_main_category_id'] = preds_main
    results_df['pred_sub_category_id'] = preds_sub
    results_df['pred_intervention_id'] = preds_interv
    results_df['pred_priority_id'] = preds_priority
    
    # Add correctness flags
    results_df['main_correct'] = (results_df['main_category_id'] == results_df['pred_main_category_id'])
    results_df['sub_correct'] = (results_df['sub_category_id'] == results_df['pred_sub_category_id'])
    results_df['interv_correct'] = (results_df['intervention_id'] == results_df['pred_intervention_id'])
    results_df['priority_correct'] = (results_df['priority_id'] == results_df['pred_priority_id'])
    
    results_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to: {output_path}")

def main():
    # Load configuration
    config = load_config()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load test data
    test_csv_path = config['data']['test_split_output']
    if not os.path.exists(test_csv_path):
        logger.error(f"Test data file not found: {test_csv_path}")
        logger.error("Please run the training script first to generate the test split.")
        return
    
    test_df = load_test_data(test_csv_path)
    
    # Load model
    model_output_dir = config['model']['output_dir']
    metadata_file = os.path.join(model_output_dir, "model_metadata.json")
    
    # Check if model metadata exists
    if not os.path.exists(metadata_file):
        logger.error(f"Model metadata file not found: {metadata_file}")
        logger.error("\nPossible reasons:")
        logger.error("1. Training hasn't been completed yet")
        logger.error("2. Model output directory is incorrect in config.yaml")
        logger.error("3. Training failed before saving the model")
        logger.error(f"\nExpected metadata location: {metadata_file}")
        logger.error(f"Model output directory: {model_output_dir}")
        
        # Check if directory exists
        if not os.path.exists(model_output_dir):
            logger.error(f"\n❌ Model output directory doesn't exist: {model_output_dir}")
            logger.error("Please run the training script first!")
        else:
            logger.error(f"\n✓ Model output directory exists but contains:")
            for item in os.listdir(model_output_dir):
                logger.error(f"  - {item}")
        return
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded metadata: {metadata}")
    
    model_path = os.path.join(model_output_dir, metadata['last_best_model_dir'])
    
    if not os.path.exists(model_path):
        logger.error(f"Model directory not found: {model_path}")
        logger.error("The metadata points to a model that doesn't exist.")
        return
    
    model, tokenizer = load_model_and_tokenizer(model_path)
    model = model.to(device)
    
    # Evaluate
    metrics, preds_main, preds_sub, preds_interv, preds_priority = evaluate_model(
        test_df, model, tokenizer, device
    )
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    # Save predictions
    predictions_output = test_csv_path.replace('.csv', '_predictions.csv')
    save_predictions(test_df, preds_main, preds_sub, preds_interv, preds_priority, predictions_output)
    
    # Save metrics
    metrics_output = test_csv_path.replace('.csv', '_evaluation_metrics.json')
    with open(metrics_output, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to: {metrics_output}")

if __name__ == "__main__":
    main()