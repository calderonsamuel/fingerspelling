"""
Training loop for multi-task fingerspelling model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

from ..models.multitask_model import MultiTaskFingerspellingModel
from ..training.losses import MultiTaskLoss
from ..evaluation.metrics import create_evaluator


class FingerspellingTrainer:
    """Trainer for multi-task fingerspelling model."""
    
    def __init__(
        self,
        model: MultiTaskFingerspellingModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        loss_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Multi-task model
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            device: Device to train on
            loss_weights: Weights for different loss components
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Initialize loss function
        if loss_weights is None:
            loss_weights = {
                'detection_weight': 1.0,
                'recognition_weight': 1.0,
                'ler_weight': 0.1,
                'pose_weight': 0.5
            }
        
        self.criterion = MultiTaskLoss(**loss_weights)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Initialize evaluator
        self.evaluator = create_evaluator()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_losses = {
            'total_loss': 0.0,
            'detection_loss': 0.0,
            'recognition_loss': 0.0,
            'ler_loss': 0.0,
            'pose_loss': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f"Epoch {self.epoch}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(batch['frames'])
                
                # Calculate loss
                losses = self.criterion(predictions, batch)
                
                # Backward pass
                losses['total_loss'].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Update metrics
                for key in total_losses:
                    if key in losses:
                        total_losses[key] += losses[key].item()
                
                # Update progress bar
                current_loss = losses['total_loss'].item()
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})
        
        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        
        return avg_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_losses = {
            'total_loss': 0.0,
            'detection_loss': 0.0,
            'recognition_loss': 0.0,
            'ler_loss': 0.0,
            'pose_loss': 0.0
        }
        
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                predictions = self.model(batch['frames'])
                
                # Calculate loss
                losses = self.criterion(predictions, batch)
                
                # Update metrics
                for key in total_losses:
                    if key in losses:
                        total_losses[key] += losses[key].item()
        
        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        
        # Calculate evaluation metrics
        eval_metrics = self.evaluator.evaluate_batch(predictions, batch)
        avg_losses.update(eval_metrics)
        
        return avg_losses
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                device_batch[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        device_batch[key][sub_key] = sub_value.to(self.device)
                    else:
                        device_batch[key][sub_key] = sub_value
            else:
                device_batch[key] = value
        
        return device_batch
    
    def train(
        self, 
        num_epochs: int,
        save_dir: Optional[str] = None,
        save_every: int = 5,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training history
        """
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        no_improvement_count = 0
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics)
            
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            
            # Validation
            if self.val_loader is not None:
                val_metrics = self.validate_epoch()
                self.val_losses.append(val_metrics)
                
                print(f"Val Loss: {val_metrics['total_loss']:.4f}")
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics['total_loss'])
                
                # Early stopping
                if val_metrics['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total_loss']
                    no_improvement_count = 0
                    
                    # Save best model
                    if save_dir:
                        self.save_checkpoint(save_path / "best_model.pth")
                else:
                    no_improvement_count += 1
                
                if no_improvement_count >= early_stopping_patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break
            
            # Save checkpoint
            if save_dir and (epoch + 1) % save_every == 0:
                self.save_checkpoint(save_path / f"checkpoint_epoch_{epoch + 1}.pth")
            
            print("-" * 50)
        
        # Save final model
        if save_dir:
            self.save_checkpoint(save_path / "final_model.pth")
            self.save_training_history(save_path / "training_history.json")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history
        }
    
    def save_checkpoint(self, filepath: Path) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Checkpoint loaded from {filepath}")
    
    def save_training_history(self, filepath: Path) -> None:
        """Save training history to JSON."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved to {filepath}")


def create_trainer(
    model: MultiTaskFingerspellingModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    learning_rate: float = 1e-4,
    loss_weights: Optional[Dict[str, float]] = None
) -> FingerspellingTrainer:
    """
    Create a trainer for the fingerspelling model.
    
    Args:
        model: Multi-task model
        train_loader: Training data loader
        val_loader: Validation data loader
        learning_rate: Learning rate
        loss_weights: Loss component weights
        
    Returns:
        Trainer instance
    """
    return FingerspellingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        loss_weights=loss_weights
    )
