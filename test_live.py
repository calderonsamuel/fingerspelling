"""
Simple test script for live fingerspelling inference.
This script provides an easy way to test your trained model with live video.
"""

import cv2
import sys
from pathlib import Path

def test_camera_access():
    """Test if camera is accessible."""
    print("Testing camera access...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not access camera 0")
        print("Try checking:")
        print("  - Camera permissions")
        print("  - Other applications using camera")
        print("  - Different camera ID (try --camera 1)")
        return False
    
    print("✅ Camera access successful")
    
    # Test frame capture
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        print(f"✅ Frame capture successful: {w}x{h}")
    else:
        print("❌ ERROR: Could not capture frame")
        cap.release()
        return False
    
    cap.release()
    return True

def check_model_files():
    """Check if trained model exists."""
    print("Checking model files...")
    
    checkpoint_path = Path("checkpoints/best_model.pth")
    config_path = Path("configs/train_config.yaml")
    
    if not checkpoint_path.exists():
        print(f"❌ ERROR: Model checkpoint not found: {checkpoint_path}")
        print("Make sure you've trained a model first:")
        print("  python train.py --subset-size 20 --epochs 5")
        return False
    
    if not config_path.exists():
        print(f"❌ ERROR: Config file not found: {config_path}")
        return False
    
    print("✅ Model files found")
    return True

def main():
    """Main test function."""
    print("🎯 Fingerspelling Live Inference Test")
    print("=" * 40)
    
    # Check prerequisites
    if not check_model_files():
        return False
    
    if not test_camera_access():
        return False
    
    print("\n🚀 Ready for live inference!")
    print("\nTo start live inference:")
    print("  python live_inference.py --mode webcam")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save predictions")
    print("\nExpected behavior:")
    print("  - Green box: Fingerspelling detected")
    print("  - Red box: No detection")
    print("  - Letters may show as '<blank>' initially (model needs more training)")
    
    # Ask if user wants to start
    try:
        response = input("\nStart live inference now? (y/n): ")
        if response.lower() in ['y', 'yes']:
            import subprocess
            subprocess.run([sys.executable, "live_inference.py", "--mode", "webcam"])
    except KeyboardInterrupt:
        print("\nExiting...")
    
    return True

if __name__ == "__main__":
    main()
