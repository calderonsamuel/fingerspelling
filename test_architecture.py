"""
Architecture validation script.
Tests the model with a small subset of data to ensure everything works.
"""

import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.preprocess import preprocess_data
from src.data.dataset import create_data_loaders
from src.models.multitask_model import create_model
from src.training.trainer import create_trainer


def test_architecture():
    """Test the architecture with a small subset of data."""
    print("Starting architecture validation...")
    
    # Test with subset including validation
    subset_size = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # 1. Test data preprocessing
        print("\n1. Testing data preprocessing...")
        processed_data = preprocess_data(
            dataset_root="dataset/ChicagoFSWild",
            output_root="test_processed_data",
            subset_size=subset_size
        )
        
        if not processed_data:
            print("ERROR: No data was processed!")
            return False
        
        print("Successfully processed data:")
        for partition, data in processed_data.items():
            print(f"  {partition}: {len(data)} sequences")
        
        # 2. Test data loaders
        print("\n2. Testing data loaders...")
        data_loaders = create_data_loaders(
            processed_data=processed_data,
            batch_size=4,  # Increased batch size for better training
            num_workers=0,
            image_size=(108, 108)
        )
        
        # Test loading a batch
        train_loader = data_loaders['train']
        sample_batch = next(iter(train_loader))
        
        print("Sample batch shapes:")
        print(f"  frames: {sample_batch['frames'].shape}")
        print(f"  sequence_lengths: {sample_batch['sequence_lengths'].shape}")
        
        # 3. Test model creation
        print("\n3. Testing model creation...")
        model = create_model(
            backbone_model="yolov8n.pt",
            use_pose=True,
            freeze_backbone=False
        )
        
        model = model.to(device)
        model.eval()
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model created with {param_count:,} parameters")
        
        # 4. Test forward pass
        print("\n4. Testing forward pass...")
        with torch.no_grad():
            # Move batch to device
            frames = sample_batch['frames'].to(device)
            
            # Forward pass
            outputs = model(frames)
            
            print(f"Model outputs:")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                elif isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            print(f"    {sub_key}: {sub_value.shape}")
        
        # 5. Test trainer creation
        print("\n5. Testing trainer creation...")
        trainer = create_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=data_loaders.get('dev'),
            learning_rate=1e-4
        )
        
        print("Trainer created successfully")
        
        # 6. Test single training step
        print("\n6. Testing single training step...")
        model.train()
        
        # Get a batch
        batch = next(iter(train_loader))
        
        # Move to device
        batch_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_device[key] = value.to(device)
            elif isinstance(value, dict):
                batch_device[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        batch_device[key][sub_key] = sub_value.to(device)
                    else:
                        batch_device[key][sub_key] = sub_value
            else:
                batch_device[key] = value
        
        # Forward pass
        trainer.optimizer.zero_grad()
        predictions = model(batch_device['frames'])
        
        # Calculate loss
        losses = trainer.criterion(predictions, batch_device)
        
        print(f"Loss components:")
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")
        
        # Backward pass
        losses['total_loss'].backward()
        trainer.optimizer.step()
        
        print("Single training step completed successfully!")
        
        # 7. Test training epochs with validation
        print("\n7. Testing training epochs with validation...")
        history = trainer.train(
            num_epochs=5,
            save_dir=None,  # Don't save during testing
            early_stopping_patience=100  # Disable early stopping
        )
        
        print(f"Training completed! Final loss: {history['train_losses'][-1]['total_loss']:.4f}")
        
        print("\n✅ Architecture validation PASSED!")
        print("All components are working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Architecture validation FAILED!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_architecture()
    sys.exit(0 if success else 1)
