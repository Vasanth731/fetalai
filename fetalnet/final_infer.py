import torch
import cv2
import numpy as np
from model import YNet
import os
from tqdm import tqdm
from skimage.morphology import skeletonize

# ---------------- CONFIG ----------------
input_channels = 1
output_channels = 64
n_class = 1
target_size = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
model = YNet(input_channels=input_channels, output_channels=output_channels, n_class=n_class)
model.load_state_dict(torch.load("/home/htic/vasanth/bio_seg/weights/fuvai_weights.pt", map_location=device))
model = model.to(device)
model.eval()  # start in eval mode


# ---------------- ENABLE DROPOUT ----------------
def enable_dropout(model):
    """Keep dropout layers active during inference."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image_for_model(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size)
    image_norm = image / 255.0
    tensor = torch.from_numpy(image_norm).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, H, W]
    return tensor.to(device), image


def preprocess_image_rgb(img_path):
    image_rgb = cv2.imread(img_path, cv2.COLOR_GRAY2RGB)
    image_rgb_resized = cv2.resize(image_rgb, target_size)
    return image_rgb, image_rgb_resized


# ---------------- INFERENCE (WITH OPTIONAL UNCERTAINTY) ----------------
@torch.no_grad()
def run_inference(image_path, monte_carlo=False, n_samples=10, save_mc_masks_dir=None, filename_prefix=""):
    input_tensor, resized_gray = preprocess_image_for_model(image_path)

    # ---------- MONTE CARLO DROPOUT MODE ----------
    if monte_carlo:
        model.train()
        enable_dropout(model)

        class_outputs = []
        seg_outputs = []

        for i in range(n_samples):
            class_output, seg_output = model(input_tensor)
            class_outputs.append(torch.softmax(class_output, dim=1).cpu().numpy())
            seg_outputs.append(seg_output.cpu().numpy())

            if save_mc_masks_dir is not None:
                seg_map_i = seg_output.cpu().numpy().squeeze()
                seg_map_uint8 = (seg_map_i * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_mc_masks_dir, f"{filename_prefix}_mc_{i+1}.png"), seg_map_uint8)

        class_outputs = np.stack(class_outputs, axis=0)
        seg_outputs = np.stack(seg_outputs, axis=0)

        mean_class_probs = np.mean(class_outputs, axis=0).squeeze()
        predicted_class = int(np.argmax(mean_class_probs))
        mean_seg_map = np.mean(seg_outputs, axis=0).squeeze()

        class_uncertainty = np.var(class_outputs, axis=0).squeeze()
        seg_uncertainty = np.var(seg_outputs, axis=0).squeeze()

        model.eval()
        return predicted_class, mean_class_probs, mean_seg_map, resized_gray, class_uncertainty, seg_uncertainty

    # ---------- STANDARD INFERENCE ----------
    model.eval()
    class_output, seg_output = model(input_tensor)
    class_probs = torch.softmax(class_output, dim=1).cpu().numpy().squeeze()
    predicted_class = int(np.argmax(class_probs))
    seg_map = seg_output.cpu().numpy().squeeze()
    return predicted_class, class_probs, seg_map, resized_gray, None, None


# ---------------- POSTPROCESS & VISUALIZATION ----------------
def scaling_contour(rgb_image, seg_map):
    img_h, img_w = rgb_image.shape[:2]
    mask = (seg_map * 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_h, mask_w = mask.shape[:2]
    scale_x, scale_y = img_w / mask_w, img_h / mask_h

    scaled_contours = []
    for c in contours:
        c = c.astype(np.float32)
        c[:, 0, 0] *= scale_x
        c[:, 0, 1] *= scale_y
        scaled_contours.append(c.astype(np.int32))

    img_with_contour = rgb_image.copy()
    cv2.drawContours(img_with_contour, scaled_contours, -1, (0, 255, 0), 5)
    overlay = cv2.addWeighted(rgb_image, 0.5, img_with_contour, 0.8, 0)
    return scaled_contours, overlay


def scaled_overlay(rgb_image, scaled_contours):
    overlay = np.zeros_like(rgb_image)
    cv2.fillPoly(overlay, scaled_contours, (0, 0, 255))
    binary_mask = np.zeros(rgb_image.shape[:2], np.uint8)
    cv2.fillPoly(binary_mask, scaled_contours, 255)
    blend = cv2.addWeighted(rgb_image, 1.0, overlay, 0.8, 0)
    return blend, overlay, binary_mask


def boundingbox_ellipsefit(rgb_image, contours):

    if rgb_image.shape[-1] == 4:
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2RGB)

    # height, width, channel = overlay.shape[:3]
    drawing = rgb_image.copy()
    ellipse_shape = np.zeros_like(rgb_image)  # Black image, same size

    rows = []
    ellipse = None
    endpoints_sorted = []

    for i, c in enumerate(contours):
        # color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        # cv2.drawContours(drawing, contours, i, (255,255,0)) # replace (255,255,0) with color for random colors

        if c.shape[0] > 5:
            ellipse = cv2.fitEllipse(c)
            (xc, yc), (major, minor), angle = ellipse

            if major < minor:
                major, minor = minor, major
                angle = (angle + 90) % 180

            # print(angle)

            theta = np.radians(angle)
            dx_major = (major / 2) * np.cos(theta)
            dy_major = (major / 2) * np.sin(theta)
            pt1_major = (round(xc + dx_major, 2), round(yc + dy_major, 2))
            pt2_major = (round(xc - dx_major, 2), round(yc - dy_major, 2))
            dx_minor = (minor / 2) * -np.sin(theta)
            dy_minor = (minor / 2) * np.cos(theta)
            pt1_minor = (round(xc + dx_minor, 2), round(yc + dy_minor, 2)) 
            pt2_minor = (round(xc - dx_minor, 2), round(yc - dy_minor, 2))

            # Sort and draw endpoints (optional markers)
            endpoints = [pt1_major, pt1_minor, pt2_major, pt2_minor]
            def angle_from_center(pt):
                return (np.arctan2(pt[1] - yc, pt[0] - xc) + 2 * np.pi) % (2 * np.pi)
            endpoints_sorted = sorted(endpoints, key=angle_from_center) # counter clockwise starting from right, top, left, bottom

            # Draw only ellipse on separate image
            cv2.ellipse(ellipse_shape, ellipse, (0, 255, 0), 2) # getting the ellipse shape for the later purpose to blow and overlay on the real image resolution 

            # points on the ellipse 
            for pt in endpoints_sorted:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(drawing, (x, y), 5, (0, 0, 255), -1)        # for the 720 image
                cv2.circle(ellipse_shape, (x, y), 5, (0, 0, 255), -1)  # for the later purpose to blow and overlay on the real image resolution


            # Draw ellipse and axes on full drawing for the 720 image
            cv2.ellipse(drawing, ellipse, (0, 255, 255), 2)
            cv2.line(drawing, (int(pt1_major[0]), int(pt1_major[1])), (int(pt2_major[0]), int(pt2_major[1])), (0, 255, 0), 1) # green for the major axis 
            cv2.line(drawing, (int(pt1_minor[0]), int(pt1_minor[1])), (int(pt2_minor[0]), int(pt2_minor[1])), (255, 0, 0), 1) # blue for the minor axis 

    return drawing, ellipse_shape, ellipse, endpoints_sorted


def extremafit_2pts(rgb_image, contours, binary_mask):
    drawing = rgb_image.copy()
    skeleton = skeletonize(binary_mask)
    skeleton_uint8 = (skeleton.astype(np.uint8) * 255)
    skeleton_overlay = cv2.addWeighted(rgb_image, 0.7, cv2.cvtColor(skeleton_uint8, cv2.COLOR_GRAY2BGR), 0.3, 0)

    coords = cv2.findNonZero(skeleton_uint8)
    if coords is None:
        return drawing, np.zeros_like(rgb_image), [], skeleton_overlay

    leftmost = tuple(coords[coords[:,:,0].argmin()][0])
    rightmost = tuple(coords[coords[:,:,0].argmax()][0])
    endpoints_sorted = [rightmost, leftmost]

    cv2.drawMarker(drawing, leftmost, (0, 255, 255), cv2.MARKER_CROSS, 30, 5)
    cv2.drawMarker(drawing, rightmost, (0, 255, 255), cv2.MARKER_CROSS, 30, 5)

    return drawing, np.zeros_like(rgb_image), endpoints_sorted, skeleton_overlay


# ---------------- PROCESS IMAGE ----------------
def process_image(image_path, contour_dir, mask_overlay_dir, drawing_dir, shape_dir, skeleton_overlay_dir, mc_masks_dir):
    filename = os.path.basename(image_path)
    filename_noext = os.path.splitext(filename)[0]

    predicted_class, class_probs, seg_map, resized_gray, class_uncert, seg_uncert = run_inference(
        image_path, monte_carlo=True, n_samples=10,
        save_mc_masks_dir=mc_masks_dir, filename_prefix=filename_noext
    )

    print(f"\nImage: {filename}")
    print("Predicted Class:", predicted_class)
    print("Class Confidence:", np.round(class_probs[predicted_class], 4))
    if class_uncert is not None:
        print("Class Uncertainty:", float(class_uncert[predicted_class]))

    image_rgb, _ = preprocess_image_rgb(image_path)
    scaled_contours, contour_overlay = scaling_contour(image_rgb, seg_map)
    cv2.imwrite(os.path.join(contour_dir, filename), contour_overlay)

    mask_overlay, overlay, binary_mask = scaled_overlay(image_rgb, scaled_contours)
    cv2.imwrite(os.path.join(mask_overlay_dir, filename), mask_overlay)
    
    drawing, shape, endpoints_sorted, skeleton_overlay = boundingbox_ellipsefit(image_rgb, scaled_contours)
    drawing, shape, endpoints_sorted, skeleton_overlay = extremafit_2pts(image_rgb, scaled_contours, binary_mask)
    print("Endpoints:", endpoints_sorted)
    cv2.imwrite(os.path.join(drawing_dir, filename), drawing)
    cv2.imwrite(os.path.join(shape_dir, filename), shape)
    cv2.imwrite(os.path.join(skeleton_overlay_dir, filename), skeleton_overlay)


# ---------------- MAIN ----------------
if __name__ == "__main__":
    input_folder = "/home/htic/vasanth/bio_seg/data_parsed/bpd"
    output_base = "/home/htic/vasanth/bio_seg/outputs/fetalnet/bpd_fullinfer"
    os.makedirs(output_base, exist_ok=True)

    contour_dir = os.path.join(output_base, "contour_overlay")
    mask_overlay_dir = os.path.join(output_base, "mask_overlay")
    drawing_dir = os.path.join(output_base, "drawing")
    shape_dir = os.path.join(output_base, "shape")
    skeleton_overlay_dir = os.path.join(output_base, "skeleton")
    mc_masks_dir = os.path.join(output_base, "mc_masks")

    for p in [contour_dir, mask_overlay_dir, drawing_dir, shape_dir, skeleton_overlay_dir, mc_masks_dir]:
        os.makedirs(p, exist_ok=True)

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    image_paths = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)])

    for img_path in tqdm(image_paths, desc="Processing images"):
        process_image(img_path, contour_dir, mask_overlay_dir, drawing_dir, shape_dir, skeleton_overlay_dir, mc_masks_dir)
