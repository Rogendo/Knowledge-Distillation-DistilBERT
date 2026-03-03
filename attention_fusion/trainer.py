"""
Multi-Task Trainer for Attention Fusion Model.

Training strategy: task alternation (better than round-robin).
Each epoch collects all batches from all 3 DataLoaders, shuffles them
together with their task tag, then iterates through in random order.
This gives the shared backbone gradients from all tasks continuously
and prevents catastrophic forgetting.
"""

import argparse
import json
import os
import random
from typing import Dict

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import yaml

from sklearn.metrics import f1_score, accuracy_score

from model import AttentionFusionModel
from datasets import (
    MultiTaskDataModule,
    QA_HEAD_CONFIG,
    build_label_maps,
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _ner_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Macro F1 over real (non-padding) NER tokens."""
    preds = logits.argmax(dim=-1).view(-1).cpu().numpy()
    true = labels.view(-1).cpu().numpy()
    mask = true != -100
    if mask.sum() == 0:
        return 0.0
    return float(f1_score(true[mask], preds[mask], average='macro', zero_division=0))


def _cls_avg_acc(logits: dict, labels: dict) -> float:
    """Average accuracy across the 4 classification heads."""
    accs = []
    for head in ('main', 'sub', 'intervention', 'priority'):
        key = f'{head}_category_id' if head in ('main', 'sub') else (
            'intervention_id' if head == 'intervention' else 'priority_id'
        )
        preds = logits[head].argmax(dim=-1).cpu().numpy()
        true = labels[key].cpu().numpy()
        mask = true != -1
        if mask.sum() == 0:
            continue
        accs.append(accuracy_score(true[mask], preds[mask]))
    return float(np.mean(accs)) if accs else 0.0


def _qa_avg_f1(logits: dict, labels: dict, threshold: float = 0.5) -> float:
    """Average micro-F1 across the 6 QA binary heads."""
    f1s = []
    for head in QA_HEAD_CONFIG:
        preds = (logits[head].sigmoid().cpu().numpy() > threshold).astype(int)
        true = labels[head].cpu().numpy().astype(int)
        f1s.append(f1_score(true, preds, average='micro', zero_division=0))
    return float(np.mean(f1s)) if f1s else 0.0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MultiTaskTrainer:
    """
    Task-alternation multi-task trainer.

    Args:
        model:          AttentionFusionModel
        train_loaders:  dict of {task: DataLoader}
        val_loaders:    dict of {task: DataLoader}
        config:         full config dict
        label_maps:     label map dict (for checkpointing)
        use_mlflow:     whether to log metrics to MLflow
    """

    def __init__(
        self,
        model: AttentionFusionModel,
        train_loaders: dict,
        val_loaders: dict,
        config: dict,
        label_maps: dict,
        use_mlflow: bool = False,
    ):
        self.model = model
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.config = config
        self.label_maps = label_maps
        self.use_mlflow = use_mlflow

        train_cfg = config['training']
        self.num_epochs = train_cfg['num_epochs']
        self.max_grad_norm = train_cfg['max_grad_norm']
        self.patience = train_cfg['early_stopping_patience']
        self.batch_size = train_cfg['batch_size']

        out_cfg = config['output']
        self.model_dir = out_cfg['model_dir']
        self.label_maps_file = out_cfg['label_maps_file']
        self.metrics_file = out_cfg['metrics_file']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay'],
        )

        # Scheduler — total steps from all loaders combined
        total_batches = sum(len(dl) for dl in train_loaders.values())
        total_steps = total_batches * self.num_epochs
        warmup_steps = int(total_steps * train_cfg['warmup_ratio'])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.best_metric = -float('inf')
        self.no_improve = 0
        self.history = []

        self._mlflow = None
        self._filter_stats = {}   # set externally from dm.filter_stats before train()
        # MLflow is initialised lazily from main() after filter_stats are available

    # ------------------------------------------------------------------

    def _setup_mlflow(self):
        """
        Configure MLflow tracking.

        Falls back to a local ./mlruns directory if the configured tracking
        server is unreachable, so training is never blocked by a missing server.
        """
        import mlflow

        mlflow_cfg = self.config.get('mlflow', {})
        tracking_uri = mlflow_cfg.get('tracking_uri', 'file:./mlruns')
        experiment_name = mlflow_cfg.get('experiment_name', 'attention_fusion_multitask')
        run_name = mlflow_cfg.get('run_name', 'multitask_train')

        # Attempt to connect to the configured URI; fall back to local store
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
        except Exception:
            print(f"  [MLflow] Could not reach {tracking_uri}, falling back to local ./mlruns")
            mlflow.set_tracking_uri('file:./mlruns')
            mlflow.set_experiment(experiment_name)

        run = mlflow.start_run(run_name=run_name)
        self._mlflow = mlflow
        self._mlflow_run_id = run.info.run_id

        # --- Log all hyperparameters as flat key=value params ---
        train_cfg = self.config['training']
        params = {
            'backbone':             self.config['backbone']['model_name'],
            'dropout':              self.config['backbone']['dropout'],
            'max_length':           self.config['tokenizer']['max_length'],
            'batch_size':           train_cfg['batch_size'],
            'learning_rate':        train_cfg['learning_rate'],
            'weight_decay':         train_cfg['weight_decay'],
            'warmup_ratio':         train_cfg['warmup_ratio'],
            'max_grad_norm':        train_cfg['max_grad_norm'],
            'num_epochs':           train_cfg['num_epochs'],
            'early_stopping_patience': train_cfg['early_stopping_patience'],
        }
        mlflow.log_params(params)

        # --- Tags: device, param count, task list ---
        total_params = sum(p.numel() for p in self.model.parameters())
        mlflow.set_tags({
            'device':       str(self.device),
            'total_params': f'{total_params:,}',
            'tasks':        'ner,cls,qa',
            'strategy':     'task_alternation',
        })

        # Log filter stats as params if available (set via dm.filter_stats before train())
        if self._filter_stats:
            filter_params = {}
            for task, stats in self._filter_stats.items():
                filter_params[f'filter_{task}_before']   = stats['before']
                filter_params[f'filter_{task}_after']    = stats['after']
                filter_params[f'filter_{task}_removed']  = stats['removed']
                filter_params[f'filter_{task}_pct_kept'] = stats['pct_kept']
            mlflow.log_params(filter_params)

        print(f"  [MLflow] Run started: {run_name} (id={self._mlflow_run_id})")
        print(f"  [MLflow] Tracking URI: {mlflow.get_tracking_uri()}")

    # ------------------------------------------------------------------

    def train_epoch(self) -> tuple:
        """
        One epoch of task-alternation training.

        Collects all batches from all loaders, shuffles them (with task tag),
        then iterates through the shuffled list — giving each task equal and
        interleaved gradient updates on the shared backbone.

        Returns:
            (avg_total_loss, {task: avg_task_loss, ...})
        """
        self.model.train()

        # Tag every batch with its task name
        all_batches = []
        for task, loader in self.train_loaders.items():
            for batch in loader:
                all_batches.append((task, batch))
        random.shuffle(all_batches)

        total_loss = 0.0
        task_losses: Dict[str, float] = {t: 0.0 for t in self.train_loaders}
        task_counts: Dict[str, int] = {t: 0 for t in self.train_loaders}

        pbar = tqdm(all_batches, desc='  Training', leave=False)
        for task, batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = _move_labels(batch['labels'], self.device)

            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task=task,
                labels=labels,
            )
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val
            task_losses[task] += loss_val
            task_counts[task] += 1
            pbar.set_postfix({t: f"{task_losses[t]/max(task_counts[t],1):.4f}" for t in task_losses})

        avg_loss = total_loss / len(all_batches)
        per_task_avg = {t: task_losses[t] / max(task_counts[t], 1) for t in task_losses}

        print(f"  Train loss: {avg_loss:.4f}  "
              + "  ".join(f"{t}={per_task_avg[t]:.4f}" for t in per_task_avg))
        return avg_loss, per_task_avg

    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict:
        """
        Per-task validation.

        Returns:
            {
              "ner_f1":      float,
              "cls_avg_acc": float,
              "qa_avg_f1":   float,
            }
        """
        self.model.eval()
        metrics = {}

        # NER
        ner_f1s = []
        for batch in self.val_loaders['ner']:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, task='ner')
            ner_f1s.append(_ner_f1(out['logits'], labels))
        metrics['ner_f1'] = float(np.mean(ner_f1s)) if ner_f1s else 0.0

        # Classification
        cls_accs = []
        for batch in self.val_loaders['cls']:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = _move_labels(batch['labels'], self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, task='cls')
            cls_accs.append(_cls_avg_acc(out['logits'], labels))
        metrics['cls_avg_acc'] = float(np.mean(cls_accs)) if cls_accs else 0.0

        # QA
        qa_f1s = []
        for batch in self.val_loaders['qa']:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = _move_labels(batch['labels'], self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, task='qa')
            qa_f1s.append(_qa_avg_f1(out['logits'], labels))
        metrics['qa_avg_f1'] = float(np.mean(qa_f1s)) if qa_f1s else 0.0

        return metrics

    # ------------------------------------------------------------------

    def save_checkpoint(self, metrics: dict):
        """Save model weights, label maps, and metrics. Upload artifacts to MLflow."""
        os.makedirs(self.model_dir, exist_ok=True)

        model_path = os.path.join(self.model_dir, 'model.pt')
        maps_path = os.path.join(self.model_dir, self.label_maps_file)
        metrics_path = os.path.join(self.model_dir, self.metrics_file)

        torch.save(self.model.state_dict(), model_path)

        with open(maps_path, 'w') as f:
            json.dump(self.label_maps, f, indent=2)

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Upload artifacts to MLflow so the best checkpoint is tracked in the run
        if self._mlflow:
            self._mlflow.log_artifact(model_path, artifact_path='checkpoint')
            self._mlflow.log_artifact(maps_path, artifact_path='checkpoint')
            self._mlflow.log_artifact(metrics_path, artifact_path='checkpoint')

        print(f"  Checkpoint saved to {self.model_dir}")

    # ------------------------------------------------------------------

    def train(self):
        """Full training loop with early stopping on average val metric."""
        print(f"Training on {self.device}")
        print(f"Epochs: {self.num_epochs}, patience: {self.patience}")

        stopped_early = False

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            train_loss, per_task_loss = self.train_epoch()
            val_metrics = self.evaluate()

            avg_val = float(np.mean(list(val_metrics.values())))
            val_metrics['avg'] = avg_val
            val_metrics['train_loss'] = train_loss
            self.history.append(val_metrics)

            print(f"  Val: ner_f1={val_metrics['ner_f1']:.4f}  "
                  f"cls_avg_acc={val_metrics['cls_avg_acc']:.4f}  "
                  f"qa_avg_f1={val_metrics['qa_avg_f1']:.4f}  "
                  f"avg={avg_val:.4f}")

            # Log all metrics to MLflow for this epoch
            if self._mlflow:
                mlflow_metrics = {
                    # Training losses
                    'train/loss':        train_loss,
                    'train/loss_ner':    per_task_loss.get('ner', 0.0),
                    'train/loss_cls':    per_task_loss.get('cls', 0.0),
                    'train/loss_qa':     per_task_loss.get('qa', 0.0),
                    # Validation metrics
                    'val/ner_f1':        val_metrics['ner_f1'],
                    'val/cls_avg_acc':   val_metrics['cls_avg_acc'],
                    'val/qa_avg_f1':     val_metrics['qa_avg_f1'],
                    'val/avg':           avg_val,
                    # Early stopping progress
                    'early_stop/no_improve': self.no_improve,
                    'early_stop/best_avg':   self.best_metric,
                }
                self._mlflow.log_metrics(mlflow_metrics, step=epoch)

            if avg_val > self.best_metric:
                self.best_metric = avg_val
                self.no_improve = 0
                self.save_checkpoint(val_metrics)
                if self._mlflow:
                    self._mlflow.log_metric('best/avg_val', avg_val, step=epoch)
                print(f"  New best avg val metric: {avg_val:.4f}")
            else:
                self.no_improve += 1
                print(f"  No improvement ({self.no_improve}/{self.patience})")
                if self.no_improve >= self.patience:
                    print("Early stopping triggered.")
                    stopped_early = True
                    break

        # Finalise the MLflow run
        if self._mlflow:
            self._mlflow.set_tags({
                'stopped_early':   str(stopped_early),
                'best_avg_val':    f'{self.best_metric:.4f}',
                'epochs_completed': str(len(self.history)),
            })
            self._mlflow.end_run()
            print(f"  [MLflow] Run finalised.")

        print(f"\nTraining complete. Best avg val metric: {self.best_metric:.4f}")
        return self.history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _move_labels(labels, device):
    """Move label tensors (or dicts of tensors) to device."""
    if isinstance(labels, torch.Tensor):
        return labels.to(device)
    return {k: v.to(device) for k, v in labels.items()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve paths relative to the config file's directory
    config_dir = os.path.dirname(os.path.abspath(config_path))
    for key in ('ner_path', 'classification_path', 'qa_path'):
        raw = config['data'][key]
        if not os.path.isabs(raw):
            config['data'][key] = os.path.normpath(os.path.join(config_dir, raw))

    # Build label maps
    label_maps_path = os.path.join(
        config['output']['model_dir'],
        config['output']['label_maps_file'],
    )
    label_maps = build_label_maps(
        ner_path=config['data']['ner_path'],
        cls_path=config['data']['classification_path'],
        qa_path=config['data']['qa_path'],
        output_path=label_maps_path,
    )

    cls_maps = label_maps['cls']
    num_main = len(cls_maps['main_cat2id'])
    num_sub = len(cls_maps['sub_cat2id'])
    num_interv = len(cls_maps['interv2id'])
    num_priority = len(cls_maps['priority2id'])

    print(f"Label counts — main: {num_main}, sub: {num_sub}, "
          f"interv: {num_interv}, priority: {num_priority}")

    # Tokenizer
    backbone = config['backbone']['model_name']
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    # Save tokenizer to model dir so inference can load it
    os.makedirs(config['output']['model_dir'], exist_ok=True)
    tokenizer.save_pretrained(config['output']['model_dir'])

    # Data
    dm = MultiTaskDataModule(config, tokenizer, label_maps)
    dm.setup()

    train_loaders = dm.get_train_loaders(config['training']['batch_size'])
    val_loaders = dm.get_val_loaders(config['training']['batch_size'])

    # Model
    model = AttentionFusionModel(
        backbone_name=backbone,
        dropout=config['backbone']['dropout'],
        num_ner_labels=len(label_maps['ner']['label_to_id']),
        num_main_cat=num_main,
        num_sub_cat=num_sub,
        num_intervention=num_interv,
        num_priority=num_priority,
        qa_heads_config=label_maps['qa']['head_config'],
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train
    use_mlflow = config.get('mlflow', {}).get('enabled', False)
    trainer = MultiTaskTrainer(
        model=model,
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        config=config,
        label_maps=label_maps,
        use_mlflow=use_mlflow,
    )
    # Pass filter stats then initialise MLflow so drop counts are logged as params
    trainer._filter_stats = getattr(dm, 'filter_stats', {})
    if use_mlflow:
        trainer._setup_mlflow()
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Attention Fusion Multi-Task Model')
    parser.add_argument('--config', default='config.yaml', help='Path to config YAML')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        exit(1)

    main(args.config)
