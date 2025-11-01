
import torch
import numpy as np
import cv2
import os
from model_with_dropout import UNET
from utils import load_checkpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm
import json
from skimage.morphology import skeletonize

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "weights/fl.pth.tar"
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
INPUT_DIR = "/home/htic/fetalai_data/vasanth/new_parsed_dataset/03_Femur_Length"
OUTPUT_DIR = "/home/htic/vasanth/bio_seg/outputs/unet_seg/new_parsed_data/fl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- MONTE CARLO DROPOUT ----------------
def enable_dropout(model):
    """Enable dropout layers during inference for Monte Carlo dropout"""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout2d) or isinstance(module, torch.nn.Dropout):
            module.train()

def monte_carlo_inference(model, x, n_samples=10):
    """
    Perform Monte Carlo dropout inference
    Args:
        model: The neural network model
        x: Input tensor
        n_samples: Number of Monte Carlo samples
    Returns:
        mean_pred: Mean prediction across all samples
        std_pred: Standard deviation (uncertainty) across all samples
    """
    model.eval()  # Set model to eval mode first
    enable_dropout(model)  # Then enable dropout layers

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = torch.sigmoid(model(x))
            predictions.append(pred.cpu().numpy())

    predictions = np.array(predictions)  # Shape: (n_samples, 1, H, W)
    mean_pred = np.mean(predictions, axis=0)  # Mean across samples
    std_pred = np.std(predictions, axis=0)    # Std deviation (uncertainty)

    return mean_pred, std_pred

# ---------------- TRANSFORM ----------------
val_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

# ---------------- ELLIPSE FIT FUNCTION ----------------
def boundingbox_ellipsefit(rgb_image, contour):
    if rgb_image is None:
        raise ValueError("rgb_image is None")

    if rgb_image.shape[-1] == 4:
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2RGB)

    drawing = rgb_image.copy()
    ellipse_img = np.zeros_like(rgb_image)
    ellipse_params = None
    endpoints_sorted = []

    cnt = np.asarray(contour)
    if cnt.ndim == 3 and cnt.shape[1] == 1:
        cnt = cnt.reshape(-1, 2)

    if cnt.shape[0] > 5:
        try:
            ellipse = cv2.fitEllipse(cnt)
        except cv2.error:
            return ellipse_img, drawing, None, []

        (xc, yc), (major, minor), angle = ellipse

        if major < minor:
            major, minor = minor, major
            angle = (angle + 90) % 180

        theta = np.radians(angle)
        dx_major = (major / 2) * np.cos(theta)
        dy_major = (major / 2) * np.sin(theta)
        pt1_major = (xc + dx_major, yc + dy_major)
        pt2_major = (xc - dx_major, yc - dy_major)
        dx_minor = (minor / 2) * -np.sin(theta)
        dy_minor = (minor / 2) * np.cos(theta)
        pt1_minor = (xc + dx_minor, yc + dy_minor)
        pt2_minor = (xc - dx_minor, yc - dy_minor)
        endpoints = [pt1_major, pt1_minor, pt2_major, pt2_minor]

        def angle_from_center(pt):
            return (np.arctan2(pt[1] - yc, pt[0] - xc) + 2 * np.pi) % (2 * np.pi)

        endpoints_sorted = sorted(endpoints, key=angle_from_center)
        cv2.ellipse(
            ellipse_img,
            ((int(round(xc)), int(round(yc))), (int(round(major)), int(round(minor))), float(angle)),
            (0, 255, 0),
            2,
        )
        for pt in endpoints_sorted:
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.drawMarker(drawing, (x, y), (0, 255, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=25, thickness=3, line_type=cv2.LINE_AA)
            cv2.circle(drawing, (x, y), 4, (0, 0, 255), -1)

        ellipse_params = {
            "center": (float(xc), float(yc)),
            "axes": (float(major), float(minor)),
            "angle": float(angle)
        }

    return ellipse_img, drawing, ellipse_params, endpoints_sorted


def extremafit_2pts(rgb_image, contour, binary_mask):
    drawing = rgb_image.copy()
    height, width = rgb_image.shape[:2]
    line_mask = np.zeros((height, width, 3), dtype=np.uint8)
    contour_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)

    binary_mask_bool = contour_mask > 0
    skeleton = skeletonize(binary_mask_bool)
    skeleton_uint8 = (skeleton.astype(np.uint8) * 255)
    skeleton_color = np.zeros_like(rgb_image)
    skeleton_color[skeleton_uint8 > 0] = (0, 0, 255)
    skeleton_overlay = cv2.addWeighted(rgb_image, 0.7, skeleton_color, 0.3, 0)

    coords = cv2.findNonZero(skeleton_uint8)
    if coords is None or len(coords) < 2:
        return drawing, line_mask, [], skeleton_overlay

    xs = coords[:, 0, 0]
    leftmost = tuple(coords[np.argmin(xs), 0])
    rightmost = tuple(coords[np.argmax(xs), 0])
    extrema_points = [leftmost, rightmost]

    for pt in extrema_points:
        cv2.drawMarker(drawing, pt, (0, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=30, thickness=4, line_type=cv2.LINE_AA)
        cv2.circle(line_mask, pt, 10, (0, 0, 255), -1)

    cv2.line(line_mask, leftmost, rightmost, (0, 255, 0), 2)

    return drawing, line_mask, extrema_points, skeleton_overlay


# ---------------- INFERENCE FUNCTION ----------------
def infer_single_image(image_path, model, threshold=0.5, n_mc_samples=10):
    """
    Perform inference with Monte Carlo dropout for uncertainty estimation
    Args:
        image_path: Path to input image
        model: The UNET model with dropout
        threshold: Threshold for binary mask generation
        n_mc_samples: Number of Monte Carlo samples for uncertainty estimation
    """
    image = np.array(Image.open(image_path).convert("RGB"))
    augmented = val_transform(image=image)
    x = augmented["image"].unsqueeze(0).to(DEVICE)

    # Perform Monte Carlo dropout inference
    mean_pred, std_pred = monte_carlo_inference(model, x, n_samples=n_mc_samples)

    # Extract mean and std predictions
    conf_mask = mean_pred.squeeze()  # Shape: (H, W)
    uncertainty_mask = std_pred.squeeze()  # Shape: (H, W)

    # Create binary mask
    binary_mask = (conf_mask > threshold).astype(np.uint8)

    # Resize to original image dimensions
    h, w = image.shape[:2]
    conf_mask_resized = cv2.resize(conf_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    uncertainty_mask_resized = cv2.resize(uncertainty_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    binary_mask_resized = cv2.resize(binary_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    mask_0_255 = (binary_mask_resized * 255).astype(np.uint8)

    # Calculate segmentation confidence score (mean probability in segmented region)
    if binary_mask_resized.sum() > 0:
        seg_confidence = conf_mask_resized[binary_mask_resized > 0].mean()
    else:
        seg_confidence = 0.0

    # Calculate uncertainty score (mean std deviation in segmented region)
    if binary_mask_resized.sum() > 0:
        uncertainty_score = uncertainty_mask_resized[binary_mask_resized > 0].mean()
    else:
        uncertainty_score = 0.0

    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"  → Segmentation Confidence: {seg_confidence:.4f}")
    print(f"  → Uncertainty Score: {uncertainty_score:.4f}")

    # Find contours
    contours_info = cv2.findContours(mask_0_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_info) == 3:
        _, contours, _ = contours_info
    else:
        contours, _ = contours_info

    # Create filename with both confidence and uncertainty scores
    base = os.path.splitext(os.path.basename(image_path))[0]
    filename_base = f"{base}_conf-{seg_confidence:.4f}_unc-{uncertainty_score:.4f}"

    if len(contours) == 0:
        print(" → No contours found.")
        conf_heatmap = (conf_mask_resized * 255).astype(np.uint8)
        conf_heatmap_color = cv2.applyColorMap(conf_heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.7, conf_heatmap_color, 0.3, 0)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{filename_base}_mask.png"), mask_0_255)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{filename_base}_overlay.png"), overlay)
        return

    largest = max(contours, key=cv2.contourArea)

    # Generate arrowhead markings (extrema points) and other visualizations
    extrema_img, line_mask, extrema_points, skeleton_overlay = extremafit_2pts(image, largest, mask_0_255)

    # Create contour visualization
    contour_img = image.copy()
    cv2.drawContours(contour_img, [largest], -1, (0, 255, 255), 3)

    # Create confidence heatmap overlay
    conf_heatmap = (conf_mask_resized * 255).astype(np.uint8)
    conf_heatmap_color = cv2.applyColorMap(conf_heatmap, cv2.COLORMAP_JET)
    overlay_bgr = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.7, conf_heatmap_color, 0.3, 0)

    # ---------------- SAVE ALL OUTPUTS ----------------
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{filename_base}_mask.png"), mask_0_255)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{filename_base}_overlay.png"), overlay_bgr)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{filename_base}_contour.png"), cv2.cvtColor(contour_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{filename_base}_arrowhead.png"), cv2.cvtColor(extrema_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{filename_base}_line_mask.png"), line_mask)

    print(f" → Saved all outputs (Conf: {seg_confidence:.4f}, Unc: {uncertainty_score:.4f})")


# ---------------- MAIN ----------------
def main():
    model = UNET(in_channels=3, out_channels=1, dropout_prob=0.5).to(DEVICE)
    load_checkpoint(torch.load(CHECKPOINT_PATH, map_location=DEVICE), model)
    model.eval()

    image_files = [f for f in os.listdir(INPUT_DIR)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    print(f"Found {len(image_files)} images in {INPUT_DIR}")
    for img_name in tqdm(image_files, desc="Running Inference"):
        infer_single_image(os.path.join(INPUT_DIR, img_name), model)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
