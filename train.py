"""
Main training script for fingerspelling detection and recognition.
"""

import torch
from pathlib import Path
import argparse
import yaml
from typing import Dict, Any

from src.data.preprocess import preprocess_data
from src.data.dataset import create_data_loaders
from src.models.multitask_model import create_model
from src.training.trainer import create_trainer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train fingerspelling model')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--subset-size', type=int, default=None,
                       help='Size of subset for testing (None for full dataset)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--force-split', action='store_true',
                       help='Force train/val split even with subset')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Preprocess data
    print("Preprocessing data...")
    processed_data = preprocess_data(
        dataset_root=config['data']['dataset_root'],
        output_root=config['data']['output_root'],
        subset_size=args.subset_size
    )
    
    if not processed_data:
        print("ERROR: No data was processed!")
        return
    
    print("Successfully processed data:")
    for partition, data in processed_data.items():
        print(f"  {partition}: {len(data)} sequences")
    
    # Create data loaders
    print("Creating data loaders...")
    data_loaders = create_data_loaders(
        processed_data=processed_data,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        image_size=tuple(config['data']['image_size'])
    )
    
    # Verify we have training data
    if 'train' not in data_loaders:
        print("ERROR: No training data available!")
        return
    
    # Create model
    print("Creating model...")
    model = create_model(
        backbone_model=config['model']['backbone'],
        use_pose=config['model']['use_pose'],
        freeze_backbone=config['model']['freeze_backbone']
    )
    
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    print("Creating trainer...")
    trainer = create_trainer(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders.get('dev'),
        learning_rate=config['training']['learning_rate'],
        loss_weights=config['training'].get('loss_weights')
    )
    
    # Print training setup
    val_info = f" and {len(data_loaders['dev'])} validation batches" if 'dev' in data_loaders else ""
    print(f"Training setup: {len(data_loaders['train'])} training batches{val_info}")
    
    # Train model
    print("Starting training...")
    num_epochs = args.epochs if args.epochs is not None else config['training']['num_epochs']
    print(f"Training for {num_epochs} epochs...")
    
    history = trainer.train(
        num_epochs=num_epochs,
        save_dir=config['training']['save_dir'],
        save_every=config['training'].get('save_every', 5),
        early_stopping_patience=config['training'].get('early_stopping_patience', 10)
    )
    
    print("Training completed!")
    if history and 'train_losses' in history and len(history['train_losses']) > 0:
        final_loss = history['train_losses'][-1]
        if isinstance(final_loss, dict) and 'total_loss' in final_loss:
            print(f"Final training loss: {final_loss['total_loss']:.4f}")
        else:
            print(f"Final training loss: {final_loss:.4f}")
    
    if trainer.best_val_loss is not None:
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    else:
        print("No validation data was used.")


if __name__ == "__main__":
    main()
