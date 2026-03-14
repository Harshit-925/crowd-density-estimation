import os
import glob
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import h5py
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from crowd_density_estimation_v2 import CrowdDensityCNN

TEST_IMG_DIR = r"C:\Users\hk672\Downloads\crowd_project\ShanghaiTech\part_A\test_data\images"
TEST_GT_DIR  = r"C:\Users\hk672\Downloads\crowd_project\ShanghaiTech\part_A\test_data\ground-truth-h5"
MODEL_PATH   = r"C:\Users\hk672\Downloads\crowd_project\outputs\best_model.pth"

DEVICE = torch.device("cpu")
model  = CrowdDensityCNN(load_weights=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()
print("Model loaded!")

transform = transforms.Compose([
    transforms.Resize((384, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_density_map(h5_path):
    with h5py.File(h5_path, "r") as f:
        density = np.array(f["density"], dtype=np.float32)
    return density


def get_h5_path(img_path):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(TEST_GT_DIR, stem + ".h5")


def get_accuracy(gt_count, pred_count):
    if gt_count == 0:
        return 0.0
    return max(0, (1 - abs(gt_count - pred_count) / gt_count) * 100)


def compare(image_path):
    img     = Image.open(image_path).convert("RGB")
    tensor  = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    pred_count = output.sum().item()

    # Load ground truth density map
    h5_path = get_h5_path(image_path)
    gt_density = load_density_map(h5_path)
    gt_count   = float(gt_density.sum())

    # Resize density maps for display
    gt_display = cv2.resize(gt_density, (512, 384), interpolation=cv2.INTER_LINEAR)

    pred_up = F.interpolate(
        output, scale_factor=8, mode="bilinear", align_corners=False
    ).squeeze().numpy()

    # De-normalise image for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    disp = (tensor.squeeze() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    error    = abs(gt_count - pred_count)
    accuracy = get_accuracy(gt_count, pred_count)

    # Determine accuracy label and color
    if accuracy >= 80:
        acc_color = "green"
        acc_label = "Good"
    elif accuracy >= 60:
        acc_color = "orange"
        acc_label = "Moderate"
    else:
        acc_color = "red"
        acc_label = "Poor"

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")

    # Panel 1 — Original Image
    axes[0].imshow(disp)
    axes[0].set_title("Original Image", fontsize=13, color="white", pad=10)
    axes[0].axis("off")

    # Panel 2 — Ground Truth
    im1 = axes[1].imshow(gt_display, cmap="jet")
    axes[1].set_title(f"Ground Truth\nActual Count: {gt_count:.0f} people",
                      fontsize=13, color="white", pad=10)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3 — Prediction
    im2 = axes[2].imshow(pred_up, cmap="jet")
    axes[2].set_title(f"Predicted\nPredicted Count: {pred_count:.0f} people",
                      fontsize=13, color="white", pad=10)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Main title with all stats
    fig.suptitle(
        f"Image: {os.path.basename(image_path)}   |   "
        f"Actual: {gt_count:.0f}   |   "
        f"Predicted: {pred_count:.0f}   |   "
        f"Error: {error:.0f}   |   "
        f"Accuracy: {accuracy:.1f}%  ({acc_label})",
        fontsize=13, fontweight="bold", color=acc_color, y=1.01
    )

    plt.tight_layout()

    save_path = "compare_result.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    # Print summary
    print(f"\n{'='*45}")
    print(f"  Image     : {os.path.basename(image_path)}")
    print(f"  Actual    : {gt_count:.0f} people")
    print(f"  Predicted : {pred_count:.0f} people")
    print(f"  Error     : {error:.0f} people")
    print(f"  Accuracy  : {accuracy:.1f}%  ({acc_label})")
    print(f"{'='*45}")
    print(f"  Saved     : {save_path}")


# Pick random test image
all_images = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))

if len(all_images) == 0:
    print("No images found! Check TEST_IMG_DIR path.")
else:
    random_image = random.choice(all_images)
    print(f"\nRandomly picked: {os.path.basename(random_image)}")
    compare(random_image)