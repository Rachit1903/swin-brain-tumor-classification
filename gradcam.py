"""Grad-CAM for the fine-tuned Swin-Tiny brain tumor classifier.

Grad-CAM is computed on the output of the final LayerNorm (`model.norm`, a 7x7x768 feature
map), which feeds the classification head directly.

Usage
    python gradcam.py --image data/Testing/glioma/Te-gl_0041.jpg
    python gradcam.py --image path/to/mri.jpg --weights saved_models/swin_final.pth --out result.png
"""
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224
MEAN = STD = np.array([0.5, 0.5, 0.5])


def load_model(weights, device):
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model.to(device).eval()


def gradcam(model, image_path, device, target_class=None):
    """Return (rgb_image, heatmap_rgb, overlay_rgb, predicted_class, confidence)."""
    store = {}
    fwd = model.norm.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    bwd = model.norm.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))

    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    x = torch.tensor((img / 255.0 - MEAN) / STD).permute(2, 0, 1).unsqueeze(0).float().to(device)
    x.requires_grad_(True)

    logits = model(x)
    probs = logits.softmax(1)[0]
    pred = int(logits.argmax(1))
    target = pred if target_class is None else target_class
    model.zero_grad()
    logits[0, target].backward()
    fwd.remove()
    bwd.remove()

    act = store["act"].permute(0, 3, 1, 2)    # (B, H, W, C) -> (B, C, H, W)
    grad = store["grad"].permute(0, 3, 1, 2)
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(1)).squeeze()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    cam = cv2.resize(cam.detach().cpu().numpy(), (IMG_SIZE, IMG_SIZE))

    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.5 * heat + 0.5 * img)
    return img, heat, overlay, CLASSES[pred], float(probs[pred])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default="saved_models/swin_final.pth")
    ap.add_argument("--out", default="gradcam_result.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)
    img, heat, overlay, label, conf = gradcam(model, args.image, device)

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.7))
    for a, im, title in zip(ax, (img, heat, overlay), ("Original MRI", "Grad-CAM heatmap", "Overlay")):
        a.imshow(im)
        a.set_title(title)
        a.axis("off")
    fig.suptitle(f"Predicted: {label} ({conf * 100:.1f}%)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Predicted {label} ({conf:.4f}) -> {args.out}")


if __name__ == "__main__":
    main()
