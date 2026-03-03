import os
import torch
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DistilBertPreTrainedModel,
    DistilBertModel,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
import torch.nn as nn
import json
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import mlflow
import mlflow.pytorch
import logging
from sklearn.metrics import precision_recall_fscore_support

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#LOAD CONFIGURATION
def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config

# Load configuration
config = load_config()

#SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
logger.info(f"Using device: {device}")

# Setup MLflow
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

#DATA LOADING
# Handle both JSON and JSONL (newline-delimited JSON) formats
try:
    # Try reading as standard JSON first
    df = pd.read_json(config['data']['input_path'])
    logger.info(f"Successfully loaded data as standard JSON from: {config['data']['input_path']}")
except ValueError as e:
    if "Trailing data" in str(e):
        # If it fails with trailing data, try JSONL format (lines=True)
        logger.warning("Standard JSON failed, trying JSONL format (newline-delimited)...")
        df = pd.read_json(config['data']['input_path'], lines=True)
        logger.info(f"Successfully loaded data as JSONL from: {config['data']['input_path']}")
    else:
        raise e

logger.info(f"Loaded {len(df)} records from dataset")

# Sub to main category mapping
sub_to_main_mapping = {
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
    "Other": "VANE"
}

# Create new column using the mapping dictionary
df['main_category'] = df['category'].map(sub_to_main_mapping)
df['main_category'] = df['main_category'].fillna('Unknown')

#PREPROCESS LABELS
main_categories = sorted([x for x in df['main_category'].unique() if x is not None])
sub_categories = sorted([x for x in df['category'].unique() if x is not None])
interventions = sorted([x for x in df['intervention'].unique() if x is not None])
priorities = [1, 2, 3]
df.head()

print(df[['category', 'main_category']].head())
logger.info(df[['category', 'main_category']].head())

# Create mappings
main_cat2id = {cat: i for i, cat in enumerate(main_categories)}
sub_cat2id = {cat: i for i, cat in enumerate(sub_categories)}
interv2id = {interv: i for i, interv in enumerate(interventions)}
priority2id = {p: i for i, p in enumerate(priorities)}

# Apply mappings
df['main_category_id'] = df['main_category'].map(lambda x: main_cat2id.get(x, -1))
df['sub_category_id'] = df['category'].map(lambda x: sub_cat2id.get(x, -1))
df['intervention_id'] = df['intervention'].map(lambda x: interv2id.get(x, -1))
df['priority_id'] = df['priority'].map(lambda x: priority2id.get(x, -1))
df['narrative'] = df['narrative'].fillna('')
df['text'] = df['narrative']

#DATA VALIDATION & STRATIFICATION
logger.info(f"Dataset shape before filtering: {df.shape}")
logger.info(f"Columns: {df.columns.tolist()}")

# Filter out classes with less than 2 samples for stratification
valid_sub_cats = df['sub_category_id'].value_counts()
logger.info(f"Sub-category distribution:\n{valid_sub_cats}")

valid_sub_cats = valid_sub_cats[valid_sub_cats >= 2].index
df_strat = df[df['sub_category_id'].isin(valid_sub_cats)]

logger.info(f"Dataset shape after filtering (min 2 samples per class): {df_strat.shape}")
logger.info(f"Filtered out {len(df) - len(df_strat)} samples with insufficient class representation")

# Check if we have enough data
if len(df_strat) < 10:
    raise ValueError(f"Insufficient data for training. Only {len(df_strat)} samples after filtering.")

#SAVE LABEL MAPPINGS EARLY
# Save these BEFORE training so they're available even if training fails
labels_dir = config['output']['labels_dir']
os.makedirs(labels_dir, exist_ok=True)

logger.info(f"💾 Saving label mappings to: {labels_dir}")

# Save label lists
with open(os.path.join(labels_dir, config['output']['main_categories_file']), "w") as f:
    json.dump(main_categories, f, indent=2)
logger.info(f"   Saved {len(main_categories)} main categories")

with open(os.path.join(labels_dir, config['output']['sub_categories_file']), "w") as f:
    json.dump(sub_categories, f, indent=2)
logger.info(f"   Saved {len(sub_categories)} sub categories")

with open(os.path.join(labels_dir, config['output']['interventions_file']), "w") as f:
    json.dump(interventions, f, indent=2)
logger.info(f"   Saved {len(interventions)} interventions")

with open(os.path.join(labels_dir, config['output']['priorities_file']), "w") as f:
    json.dump(priorities, f, indent=2)
logger.info(f"   Saved {len(priorities)} priority levels")

# Save ID mappings for reference
label_mappings = {
    'main_cat2id': main_cat2id,
    'sub_cat2id': sub_cat2id,
    'interv2id': interv2id,
    'priority2id': priority2id,
    'id2main_cat': {v: k for k, v in main_cat2id.items()},
    'id2sub_cat': {v: k for k, v in sub_cat2id.items()},
    'id2interv': {v: k for k, v in interv2id.items()},
    'id2priority': {v: k for k, v in priority2id.items()}
}

with open(os.path.join(labels_dir, "label_mappings.json"), "w") as f:
    json.dump(label_mappings, f, indent=2)

logger.info(" Label mappings saved successfully!")
logger.info(f"   Location: {labels_dir}")

#SPLIT DATASET
train_df, test_df = train_test_split(
    df_strat, 
    test_size=config['data']['test_size'], 
    random_state=config['data']['random_seed'],
    stratify=df_strat['sub_category_id']
)

#SAVE TEST SPLIT TO CSV
test_output_path = config['data']['test_split_output']
os.makedirs(os.path.dirname(test_output_path), exist_ok=True)

# Save test data with all relevant columns
test_df.to_csv(test_output_path, index=False)
logger.info(f"Test split saved to: {test_output_path}")
logger.info(f"Test set size: {len(test_df)} samples")

# Also save a summary of the test split
test_summary = {
    'total_samples': len(test_df),
    'train_samples': len(train_df),
    'test_split_ratio': config['data']['test_size'],
    'random_seed': config['data']['random_seed'],
    'saved_at': str(datetime.datetime.now()),
    'main_category_distribution': test_df['main_category'].value_counts().to_dict(),
    'sub_category_distribution': test_df['category'].value_counts().to_dict(),
    'intervention_distribution': test_df['intervention'].value_counts().to_dict(),
    'priority_distribution': test_df['priority'].value_counts().to_dict()
}

summary_path = test_output_path.replace('.csv', '_summary.json')
with open(summary_path, 'w') as f:
    json.dump(test_summary, f, indent=4)
logger.info(f"Test split summary saved to: {summary_path}")

#CREATE DATASETS
train_dataset = Dataset.from_pandas(train_df[['text', 'main_category_id', 'sub_category_id', 'intervention_id', 'priority_id']])
test_dataset = Dataset.from_pandas(test_df[['text', 'main_category_id', 'sub_category_id', 'intervention_id', 'priority_id']])

dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

#MODEL DEFINITION
class MultiTaskDistilBert(DistilBertPreTrainedModel):
    def __init__(self, config, num_main, num_sub, num_interv, num_priority):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier_main = nn.Linear(config.dim, num_main)
        self.classifier_sub = nn.Linear(config.dim, num_sub)
        self.classifier_interv = nn.Linear(config.dim, num_interv)
        self.classifier_priority = nn.Linear(config.dim, num_priority)
        self.dropout = nn.Dropout(config.dropout)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, 
                main_category_id=None, sub_category_id=None, 
                intervention_id=None, priority_id=None, return_outputs=False):
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_state = distilbert_output.last_hidden_state 
        pooled_output = hidden_state[:, 0]                 
        pooled_output = self.pre_classifier(pooled_output) 
        pooled_output = nn.ReLU()(pooled_output)           
        pooled_output = self.dropout(pooled_output)        
        
        logits_main = self.classifier_main(pooled_output)
        logits_sub = self.classifier_sub(pooled_output)
        logits_interv = self.classifier_interv(pooled_output)
        logits_priority = self.classifier_priority(pooled_output)
        
        loss = None
        if main_category_id is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss_main = loss_fct(logits_main, main_category_id)
            loss_sub = loss_fct(logits_sub, sub_category_id)
            loss_interv = loss_fct(logits_interv, intervention_id)
            loss_priority = loss_fct(logits_priority, priority_id)
            loss = loss_main + loss_sub + loss_interv + loss_priority
        
        if loss is not None:
            return (loss, logits_main, logits_sub, logits_interv, logits_priority)
        else:
            return (logits_main, logits_sub, logits_interv, logits_priority)
    
    def get_embeddings(self, input_ids, attention_mask):
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[-1]

#MODEL LOADING/INITIALIZATION
model_output_dir = config['model']['output_dir']
metadata_file = os.path.join(model_output_dir, "model_metadata.json")
os.makedirs(model_output_dir, exist_ok=True)

# Load metadata of the last best model
if os.path.exists(metadata_file):
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
        print(metadata)
        logger.info(f"Model Version Metadata {metadata}")

    last_best_model_path = os.path.join(model_output_dir, metadata['last_best_model_dir'])
    print(last_best_model_path)
    print(f"Loading existing model from {last_best_model_path} for continuous fine-tuning.")
    logger.info(f"Loading existing model from {last_best_model_path} for continuous fine-tuning.")
   
    if os.path.exists(last_best_model_path):
        model = MultiTaskDistilBert.from_pretrained(
            last_best_model_path,
            num_main=len(main_categories),
            num_sub=len(sub_categories),
            num_interv=len(interventions),
            num_priority=len(priorities)
        )
        tokenizer = AutoTokenizer.from_pretrained(last_best_model_path)
    else:
        print(f"Warning: Last best model directory not found. Starting from base checkpoint.")
        logger.warning(f"Warning: Last best model directory not found. Starting from base checkpoint.")
        checkpoint = config['model']['checkpoint']
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = MultiTaskDistilBert.from_pretrained(
            checkpoint,
            num_main=len(main_categories),
            num_sub=len(sub_categories),
            num_interv=len(interventions),
            num_priority=len(priorities)
        )
else:
    logger.info("No existing model found. Starting from base checkpoint.")
    checkpoint = config['model']['checkpoint']
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = MultiTaskDistilBert.from_pretrained(
        checkpoint,
        num_main=len(main_categories),
        num_sub=len(sub_categories),
        num_interv=len(interventions),
        num_priority=len(priorities)
    )

#TOKENIZATION
def tokenize_function(batch):
    encoding = tokenizer(
        batch["text"], 
        padding=config['tokenizer']['padding'], 
        truncation=config['tokenizer']['truncation'],
        max_length=config['tokenizer']['max_length']
    )
    return encoding

encoded_dataset = dataset.map(tokenize_function, batched=True)
encoded_dataset.set_format("torch", columns=[
    "input_ids", "attention_mask", 
    "main_category_id", "sub_category_id", 
    "intervention_id", "priority_id"
])

#METRICS
def compute_metrics(p: EvalPrediction):
    logger.info("compute_metrics called")
    logits_main, logits_sub, logits_interv, logits_priority = p.predictions
    labels_main, labels_sub, labels_interv, labels_priority = p.label_ids

    preds_main = np.argmax(logits_main, axis=1)
    preds_sub = np.argmax(logits_sub, axis=1)
    preds_interv = np.argmax(logits_interv, axis=1)
    preds_priority = np.argmax(logits_priority, axis=1)

    metrics = {}

    def add_task_metrics(task_name, labels, preds):
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
        
        metrics[f"{task_name}_acc"] = accuracy
        metrics[f"{task_name}_precision"] = precision
        metrics[f"{task_name}_recall"] = recall
        metrics[f"{task_name}_f1"] = f1

    add_task_metrics("main", labels_main, preds_main)
    add_task_metrics("sub", labels_sub, preds_sub)
    add_task_metrics("interv", labels_interv, preds_interv)
    add_task_metrics("priority", labels_priority, preds_priority)

    avg_acc = np.mean([metrics[f"{task}_acc"] for task in ["main", "sub", "interv", "priority"]])
    avg_precision = np.mean([metrics[f"{task}_precision"] for task in ["main", "sub", "interv", "priority"]])
    avg_recall = np.mean([metrics[f"{task}_recall"] for task in ["main", "sub", "interv", "priority"]])
    avg_f1 = np.mean([metrics[f"{task}_f1"] for task in ["main", "sub", "interv", "priority"]])

    metrics["eval_avg_acc"] = avg_acc
    metrics["eval_avg_precision"] = avg_precision
    metrics["eval_avg_recall"] = avg_recall
    metrics["eval_avg_f1"] = avg_f1

    return metrics

#CUSTOM TRAINER
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = {
            "main_category_id": inputs.pop("main_category_id"),
            "sub_category_id": inputs.pop("sub_category_id"),
            "intervention_id": inputs.pop("intervention_id"),
            "priority_id": inputs.pop("priority_id")
        }
        outputs = model(**inputs, **labels, return_outputs=True)
        loss = outputs[0]
        if return_outputs:
            return (loss, *outputs[1:])
        else:
            return loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        label_keys = ["main_category_id", "sub_category_id", "intervention_id", "priority_id"]
        labels = {key: inputs.pop(key) for key in label_keys if key in inputs}

        with torch.no_grad():
            outputs = model(**inputs)
        
        if isinstance(outputs[0], torch.Tensor) and outputs[0].dim() == 0:
            loss = outputs[0]
            logits = outputs[1:]
        else:
            loss = None
            logits = outputs

        if labels:
            label_values = (labels["main_category_id"], labels["sub_category_id"],
                           labels["intervention_id"], labels["priority_id"])
        
        return (loss, logits, label_values)

#TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy=config['training']['eval_strategy'],
    learning_rate=config['training']['learning_rate'],
    per_device_train_batch_size=config['training']['per_device_train_batch_size'],
    per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
    num_train_epochs=config['training']['num_train_epochs'],
    weight_decay=config['training']['weight_decay'],
    save_strategy=config['training']['save_strategy'],
    load_best_model_at_end=config['training']['load_best_model_at_end'],
    metric_for_best_model=config['training']['metric_for_best_model'],
    greater_is_better=config['training']['greater_is_better'],
    logging_dir='./logs',
    logging_steps=config['training']['logging_steps'],
)

logger.info(f"Test set size: {len(encoded_dataset['test'])}")
logger.info(f"Train set size: {len(encoded_dataset['train'])}")

#INITIALIZE TRAINER
trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    compute_metrics=compute_metrics,
)

#TRAINING WITH MLFLOW
with mlflow.start_run(run_name=f"multitask_distilbert_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    
    # Log parameters
    mlflow.log_param("learning_rate", training_args.learning_rate)
    mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
    mlflow.log_param("num_epochs", training_args.num_train_epochs)
    mlflow.log_param("num_main_categories", len(main_categories))
    mlflow.log_param("num_sub_categories", len(sub_categories))
    mlflow.log_param("num_interventions", len(interventions))
    mlflow.log_param("num_priorities", len(priorities))
    mlflow.log_param("test_size", config['data']['test_size'])
    mlflow.log_param("random_seed", config['data']['random_seed'])
    
    # Train the model
    trainer.train()
    
    # Evaluate and log metrics
    new_metrics = trainer.evaluate(encoded_dataset["test"])
    new_avg_acc = new_metrics.get('eval_avg_acc', 0)
    
    for metric_name, metric_value in new_metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    logger.info(f"Model Performance results: {new_metrics}")
    
    # Check if model improved
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        prev_avg_acc = metadata.get('eval_avg_acc', 0)
    else:
        prev_avg_acc = -1
    
    # Save and log model if improved
    if new_avg_acc > prev_avg_acc:
        logger.info("New model performance improved! Saving new version.")
        
        version = len(os.listdir(model_output_dir)) - 1
        new_model_dir = f"multitask_distilbert_v{version}"
        new_model_path = os.path.join(model_output_dir, new_model_dir)
        
        # Save model
        trainer.save_model(new_model_path)
        tokenizer.save_pretrained(new_model_path)
        
        # ✨ IMPORTANT: Save labels WITH the model for future reference
        model_labels_dir = os.path.join(new_model_path, 'labels')
        os.makedirs(model_labels_dir, exist_ok=True)
        
        with open(os.path.join(model_labels_dir, config['output']['main_categories_file']), "w") as f:
            json.dump(main_categories, f, indent=2)
        with open(os.path.join(model_labels_dir, config['output']['sub_categories_file']), "w") as f:
            json.dump(sub_categories, f, indent=2)
        with open(os.path.join(model_labels_dir, config['output']['interventions_file']), "w") as f:
            json.dump(interventions, f, indent=2)
        with open(os.path.join(model_labels_dir, config['output']['priorities_file']), "w") as f:
            json.dump(priorities, f, indent=2)
        
        logger.info(f" Labels saved with model at: {model_labels_dir}")
        
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="multitask_distilbert_model",
            registered_model_name=config['mlflow']['registered_model_name']
        )
        
        mlflow.log_artifact(new_model_path)
        
        metadata = {
            "version": f"v{version}",
            "date_trained": str(datetime.datetime.now()),
            "eval_avg_acc": new_avg_acc,
            "last_best_model_dir": new_model_dir,
            "metrics": new_metrics,
            "config_used": config,
            "num_classes": {
                "main_categories": len(main_categories),
                "sub_categories": len(sub_categories),
                "interventions": len(interventions),
                "priorities": len(priorities)
            }
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)
        
        mlflow.log_artifact(metadata_file)
        mlflow.log_artifact(test_output_path)  # Log the test split CSV
        
        logger.info(f"Model logged and registered in MLflow as '{config['mlflow']['registered_model_name']}'")
    else:
        logger.info("Model performance did not improve. Not saving a new version.")

#GENERATE EMBEDDINGS
def generate_category_embeddings(categories, model, tokenizer, device):
    embeddings = []
    for category in categories:
        inputs = tokenizer(
            category, 
            padding=config['tokenizer']['padding'], 
            truncation=config['tokenizer']['truncation'], 
            max_length=config['tokenizer']['max_length'], 
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            emb = model.get_embeddings(**inputs).cpu().numpy()
        embeddings.append(emb[0])
    return np.array(embeddings)

main_cat_embeddings = generate_category_embeddings(main_categories, model, tokenizer, device)
sub_cat_embeddings = generate_category_embeddings(sub_categories, model, tokenizer, device)

# Save embeddings
embeddings_dir = config['model']['embeddings_dir']
# Make embeddings dir absolute if it's relative
if not os.path.isabs(embeddings_dir):
    embeddings_dir = os.path.join(model_output_dir, embeddings_dir)
os.makedirs(embeddings_dir, exist_ok=True)

np.save(os.path.join(embeddings_dir, "main_cat_embeddings.npy"), main_cat_embeddings)
np.save(os.path.join(embeddings_dir, "sub_cat_embeddings.npy"), sub_cat_embeddings)
logger.info(f"Embeddings saved to: {embeddings_dir}")

#FINAL EVALUATION
metrics = trainer.evaluate(encoded_dataset["test"])
logger.info(f"Final Model Performance results: {metrics}")

metrics_path = os.path.join(model_output_dir, config['output']['metrics_file'])
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
logger.info(f"Final metrics saved to: {metrics_path}")

logger.info("Training complete!")