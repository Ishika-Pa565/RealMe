"""
predict.py - Standalone inference engine for RealMe Deepfake Detection.
Loads EfficientNet-B0 PyTorch model and computes P(fake) in [0, 1].
"""

import argparse
import sys
import time
from pathlib import Path
from PIL import Image
import timm
import torch
import torch.nn as nn
from torchvision import transforms

# Root directory resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Standard ImageNet transform for inference
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(weights_path: str | Path | None = None, device: str = "cpu") -> nn.Module:
    """
    Loads the EfficientNet-B0 binary classifier.
    If weights_path is None or missing, returns initialized model architecture.
    """
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)

    if weights_path:
        w_path = Path(weights_path)
        if w_path.exists():
            state_dict = torch.load(w_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            return model

    default_path = PROJECT_ROOT / "models" / "best.pth"
    if default_path.exists():
        state_dict = torch.load(default_path, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def predict_image(image_input: str | Path | Image.Image, model: nn.Module, device: str = "cpu") -> float:
    """
    Takes an image file path or PIL Image object (in memory) and returns P(fake) in [0.0, 1.0].
    Never saves images to disk.
    """
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError("image_input must be a file path or PIL Image instance.")

    tensor = INFERENCE_TRANSFORMS(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        prob_fake = torch.sigmoid(logits).item()

    return float(prob_fake)


def quick_self_test():
    """Generates a synthetic image in memory and tests CPU prediction speed."""
    print("Running quick prediction self-test on CPU...")
    start_time = time.time()

    model = load_model(device="cpu")
    dummy_img = Image.new("RGB", (300, 300), color=(128, 128, 128))

    score = predict_image(dummy_img, model, device="cpu")
    elapsed = time.time() - start_time

    print(f"Self-test output: P(fake) = {score:.4f} ({score * 100:.1f}% Fake)")
    print(f"Inference time: {elapsed:.2f} seconds (< 5s target)")
    assert 0.0 <= score <= 1.0, f"Score {score} is out of [0, 1] range!"
    assert elapsed < 5.0, f"Inference took {elapsed:.2f}s, exceeding 5s target!"
    print("Self-test PASSED successfully!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealMe Deepfake Inference")
    parser.add_argument("image_path", nargs="?", type=str, help="Path to input image")
    parser.add_argument("--weights", type=str, default="models/best.pth", help="Path to model weights")
    parser.add_argument("--test", action="store_true", help="Run in-memory self-test")

    args = parser.parse_args()

    if args.test or not args.image_path:
        quick_self_test()
    else:
        model = load_model(weights_path=args.weights, device="cpu")
        score = predict_image(args.image_path, model, device="cpu")
        label = "Fake" if score >= 0.5 else "Real"
        print(f"\nResult: The image is classified as {label}")
        print(f"Fakeness Confidence Score: {score * 100:.1f}% ({score:.4f})")
