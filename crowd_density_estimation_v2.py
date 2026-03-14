import os
import glob
import numpy as np
import h5py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


BASE_DIR  = r"C:\Users\hk672\Downloads\crowd_project\ShanghaiTech\part_A"
TRAIN_IMG = os.path.join(BASE_DIR, "train_data", "images")
TRAIN_GT  = os.path.join(BASE_DIR, "train_data", "ground-truth-h5")
TEST_IMG  = os.path.join(BASE_DIR, "test_data", "images")
TEST_GT   = os.path.join(BASE_DIR, "test_data", "ground-truth-h5")

IMG_HEIGHT = 384
IMG_WIDTH  = 512
BATCH_SIZE = 4
EPOCHS     = 5
LR         = 1e-6
STEP_SIZE  = 10
GAMMA      = 0.5
SKIP_TRAINING = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def load_density_map(h5_path):
    with h5py.File(h5_path, "r") as f:
        density = np.array(f["density"], dtype=np.float32)
    return density


def find_h5_for_image(img_path, gt_dir):
    stem    = os.path.splitext(os.path.basename(img_path))[0]
    h5_path = os.path.join(gt_dir, stem + ".h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"No .h5 found for: {img_path}")
    return h5_path


class CrowdDataset(Dataset):

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, img_dir, gt_dir):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.gt_dir    = gt_dir

        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found in: {img_dir}")

        self.img_transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        h5_path  = find_h5_for_image(img_path, self.gt_dir)

        image   = Image.open(img_path).convert("RGB")
        image   = self.img_transform(image)

        density        = load_density_map(h5_path)
        orig_h, orig_w = density.shape
        target_h       = IMG_HEIGHT // 8
        target_w       = IMG_WIDTH  // 8

        scale   = (orig_h * orig_w) / (target_h * target_w)
        density = cv2.resize(density, (target_w, target_h),
                             interpolation=cv2.INTER_CUBIC) * scale
        density = torch.tensor(density, dtype=torch.float32).unsqueeze(0)

        return image, density


class CrowdDensityCNN(nn.Module):

    def __init__(self, load_weights=True):
        super().__init__()

        weight_enum    = models.VGG16_Weights.IMAGENET1K_V1 if load_weights else None
        vgg            = models.vgg16(weights=weight_enum)
        self.frontend  = nn.Sequential(*list(vgg.features.children())[:17])

        self.backend = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(128,  64, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d( 64,  64, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d( 64,   1, kernel_size=1),
        )

        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x


def train_model(model, train_loader, val_loader=None):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    best_mae     = float("inf")
    train_losses = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images  = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            preds = model(images)
            loss  = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch [{epoch}/{EPOCHS}]  Batch [{batch_idx+1}/{len(train_loader)}]  Loss: {loss.item():.6f}")

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step()

        print(f"Epoch [{epoch}/{EPOCHS}]  Avg Loss: {avg_loss:.6f}  LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loader is not None:
            mae, rmse = evaluate_model(model, val_loader, silent=True)
            print(f"           Val MAE: {mae:.2f}  RMSE: {rmse:.2f}")
            if mae < best_mae:
                best_mae  = mae
                ckpt_path = os.path.join(OUT_DIR, "best_model.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"           Saved best model (MAE={best_mae:.2f})")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, EPOCHS + 1), train_losses, marker="o", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_loss.png"), dpi=150)
    plt.close()

    return train_losses


def evaluate_model(model, data_loader, silent=False):
    model.eval()
    model.to(DEVICE)

    mae_total = 0.0
    mse_total = 0.0
    n         = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images  = images.to(DEVICE)
            targets = targets.to(DEVICE)
            preds   = model(images)

            for pred, gt in zip(preds, targets):
                error      = abs(pred.sum().item() - gt.sum().item())
                mae_total += error
                mse_total += error ** 2
                n         += 1

    mae  = mae_total / n
    rmse = (mse_total / n) ** 0.5

    if not silent:
        print(f"\n  Images evaluated : {n}")
        print(f"  MAE  : {mae:.2f}")
        print(f"  RMSE : {rmse:.2f}\n")

    return mae, rmse


def visualize_prediction(model, dataset, idx=0, save_path=None):
    model.eval()
    model.to(DEVICE)

    image_tensor, gt_tensor = dataset[idx]
    input_batch = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_tensor = model(input_batch)

    gt_up = F.interpolate(
        gt_tensor.unsqueeze(0), scale_factor=8, mode="bilinear", align_corners=False
    ).squeeze().cpu().numpy()

    pred_up = F.interpolate(
        pred_tensor.cpu(), scale_factor=8, mode="bilinear", align_corners=False
    ).squeeze().numpy()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    disp = (image_tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    gt_count   = gt_tensor.sum().item()
    pred_count = pred_tensor.sum().item()
    error      = abs(gt_count - pred_count)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(disp)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    im1 = axes[1].imshow(gt_up, cmap="jet")
    axes[1].set_title(f"Ground Truth\nCount: {gt_count:.1f}", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(pred_up, cmap="jet")
    axes[2].set_title(f"Predicted\nCount: {pred_count:.1f}", fontsize=12)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(f"Sample {idx}  |  GT: {gt_count:.1f}  Predicted: {pred_count:.1f}  Error: {error:.1f}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUT_DIR, f"prediction_{idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}  |  GT: {gt_count:.1f}  Pred: {pred_count:.1f}  Error: {error:.1f}")


if __name__ == "__main__":

    print("\nLoading datasets...")
    train_dataset = CrowdDataset(TRAIN_IMG, TRAIN_GT)
    test_dataset  = CrowdDataset(TEST_IMG,  TEST_GT)
    print(f"  Train: {len(train_dataset)}  |  Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=1,          shuffle=False, num_workers=0)

    print("\nBuilding model...")
    model = CrowdDensityCNN(load_weights=True)
    model.to(DEVICE)

    if not SKIP_TRAINING:
        print(f"\nTraining for {EPOCHS} epochs...")
        train_model(model, train_loader, val_loader=test_loader)
    else:
        print("\nSkipping training.")

    best_ckpt = os.path.join(OUT_DIR, "best_model.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE, weights_only=True))
        print(f"Loaded: {best_ckpt}")
    else:
        print("No saved model found, using current weights.")

    print("\nEvaluating...")
    mae, rmse = evaluate_model(model, test_loader)

    print("\nGenerating visualisations...")
    for idx in [0, 5, 10]:
        if idx < len(test_dataset):
            visualize_prediction(model, test_dataset, idx=idx)

    print(f"\nDone. Outputs saved to: {OUT_DIR}")
    print(f"MAE: {mae:.2f}  |  RMSE: {rmse:.2f}")