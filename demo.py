import os
import glob
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from crowd_density_estimation_v2 import CrowdDensityCNN

TEST_IMG_DIR = r"C:\Users\hk672\Downloads\crowd_project\ShanghaiTech\part_A\test_data\images"
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

def predict(image_path):
    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    count   = output.sum().item()
    imgname = os.path.basename(image_path)

    print(f"Image  : {imgname}")
    print(f"Count  : {int(count)} people")

    pred_map = F.interpolate(
        output, scale_factor=8, mode="bilinear", align_corners=False
    ).squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title(f"Input: {imgname}", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(pred_map, cmap="jet")
    axes[1].set_title(f"Predicted Count: {int(count)} people", fontsize=12)
    axes[1].axis("off")

    plt.suptitle(f"Crowd Density Estimation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("demo_result.png", dpi=150)
    plt.close()
    print("Saved demo_result.png")


all_images = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))

if len(all_images) == 0:
    print("No images found! Check TEST_IMG_DIR path.")
else:
    random_image = random.choice(all_images)
    print(f"\nRandomly picked: {os.path.basename(random_image)}")
    predict(random_image)