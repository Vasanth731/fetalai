# import torch
# import cv2
# import numpy as np
# from model import YNet 
# import os
# from tqdm import tqdm
# from skimage.morphology import skeletonize

# # ---------------- CONFIG ----------------
# input_channels = 1
# output_channels = 64
# n_class = 1
# target_size = (224, 224)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ---------------- LOAD MODEL ----------------
# model = YNet(input_channels=input_channels, output_channels=output_channels, n_class=n_class)
# model.load_state_dict(torch.load("/home/htic/vasanth/bio_seg/weights/fuvai_weights.pt", map_location=device))
# model = model.to(device)
# model.eval()

# # ---------------- ENABLE DROPOUT ----------------
# def enable_dropout(model):
#     """Keep dropout layers active during inference."""
#     for m in model.modules():
#         if isinstance(m, torch.nn.Dropout):
#             m.train()

# # ---------------- IMAGE PREPROCESS ----------------
# def preprocess_image_for_model(img_path):
#     image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     image = cv2.resize(image, target_size)
#     image_norm = image / 255.0
#     tensor = torch.from_numpy(image_norm).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, H, W]
#     return tensor.to(device), image

# def preprocess_image_rgb(img_path):
#     image_rgb = cv2.imread(img_path, cv2.COLOR_GRAY2RGB)
#     image_rgb_resized = cv2.resize(image_rgb, target_size)
#     return image_rgb, image_rgb_resized

# # ---------------- INFERENCE ----------------
# @torch.no_grad()
# def run_inference(image_path, monte_carlo=False, n_samples=10, save_all_masks_dir=None, filename_prefix=""):
#     """
#     Run inference with optional Monte Carlo Dropout.
#     Saves each sampled segmentation mask if save_all_masks_dir is provided.
#     """
#     input_tensor, resized_gray = preprocess_image_for_model(image_path)

#     if monte_carlo:
#         enable_dropout(model)
#         class_outputs = []
#         seg_outputs = []

#         for i in range(n_samples):
#             class_output, seg_output = model(input_tensor)
#             class_outputs.append(torch.softmax(class_output, dim=1).cpu().numpy())
#             seg_outputs.append(seg_output.cpu().numpy())

#             # Save each dropout mask
#             if save_all_masks_dir is not None:
#                 seg_map_i = seg_output.cpu().numpy().squeeze()
#                 seg_map_uint8 = (np.clip(seg_map_i, 0, 1) * 255).astype(np.uint8)
#                 cv2.imwrite(os.path.join(save_all_masks_dir, f"{filename_prefix}_mask_sample_{i+1}.png"), seg_map_uint8)

#         # Average predictions
#         mean_class_probs = np.mean(class_outputs, axis=0).squeeze()
#         mean_seg_map = np.mean(seg_outputs, axis=0).squeeze()

#         predicted_class = np.argmax(mean_class_probs)

#         # Compute variance as uncertainty
#         class_uncertainty = np.var(class_outputs, axis=0).squeeze()
#         seg_uncertainty = np.var(seg_outputs, axis=0).squeeze()

#         return predicted_class, mean_class_probs, mean_seg_map, resized_gray, class_uncertainty, seg_uncertainty

#     else:
#         model.eval()
#         class_output, seg_output = model(input_tensor)
#         class_probs = torch.softmax(class_output, dim=1).cpu().numpy().squeeze()
#         predicted_class = np.argmax(class_probs)
#         seg_map = seg_output.cpu().numpy().squeeze()
#         return predicted_class, class_probs, seg_map, resized_gray, None, None

# # ---------------- POSTPROCESS & VISUALIZATION ----------------
# def scaling_contour(rgb_image, seg_map):
#     img_h, img_w = rgb_image.shape[:2]
#     mask = (seg_map * 255).astype(np.uint8)
#     mask_h, mask_w = mask.shape[:2]

#     _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     scale_x = img_w / mask_w
#     scale_y = img_h / mask_h

#     scaled_contours = []
#     for cnt in contours:
#         scaled_cnt = cnt.astype(np.float32)
#         scaled_cnt[:, 0, 0] *= scale_x
#         scaled_cnt[:, 0, 1] *= scale_y
#         scaled_contours.append(scaled_cnt.astype(np.int32))

#     img_with_contour = rgb_image.copy()
#     cv2.drawContours(img_with_contour, scaled_contours, -1, (0, 255, 0), 5)
#     contour_overlay = cv2.addWeighted(rgb_image, 0.5, img_with_contour, 0.8, 0)
#     return scaled_contours, contour_overlay

# def scaled_overlay(rgb_image, scaled_contours):
#     overlay = np.zeros_like(rgb_image, dtype=np.uint8)
#     cv2.fillPoly(overlay, scaled_contours, color=(0, 0, 255))
#     binary_mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
#     cv2.fillPoly(binary_mask, scaled_contours, color=255)
#     mask_overlay = cv2.addWeighted(rgb_image, 1.0, overlay, 0.8, 0)
#     return mask_overlay, overlay, binary_mask

# def extremafit_2pts(rgb_image, contours, binary_mask):
#     drawing = rgb_image.copy()
#     height, width = rgb_image.shape[:2]
#     line_mask = np.zeros((height, width, 3), dtype=np.uint8)
#     skeleton = skeletonize(binary_mask)
#     skeleton_uint8 = (skeleton.astype(np.uint8) * 255)
#     skeleton_color = np.zeros_like(rgb_image)
#     skeleton_color[skeleton_uint8 > 0] = (0, 0, 255)
#     skeleton_overlay = cv2.addWeighted(rgb_image, 0.7, skeleton_color, 0.3, 0)

#     coords = cv2.findNonZero(skeleton_uint8)
#     if coords is None:
#         return drawing, line_mask, [], skeleton_overlay

#     leftmost = tuple(coords[coords[:, :, 0].argmin()][0])
#     rightmost = tuple(coords[coords[:, :, 0].argmax()][0])
#     endpoints_sorted = [rightmost, leftmost]

#     cv2.line(line_mask, leftmost, rightmost, color=(0, 255, 0), thickness=2)
#     for pt in [leftmost, rightmost]:
#         cv2.drawMarker(drawing, pt, (0, 255, 255), markerType=cv2.MARKER_CROSS, 
#                        markerSize=30, thickness=5, line_type=cv2.LINE_AA)
#         cv2.circle(skeleton_overlay, pt, 5, (0, 255, 0), -1)
#     return drawing, line_mask, endpoints_sorted, skeleton_overlay

# # ---------------- PROCESS IMAGE ----------------
# def process_image(image_path, contour_dir, mask_overlay_dir, drawing_dir, shape_dir, skeleton_overlay_dir, all_masks_dir):
#     filename = os.path.basename(image_path)
#     filename_wo_ext = os.path.splitext(filename)[0]

#     predicted_class, class_probs, seg_map, resized_gray, class_uncert, seg_uncert = run_inference(
#         image_path, monte_carlo=True, n_samples=10, save_all_masks_dir=all_masks_dir, filename_prefix=filename_wo_ext
#     )

#     print(f"\nProcessing: {image_path}")
#     print("Predicted class:", predicted_class)
#     if class_uncert is not None:
#         # print(f"Class uncertainty (var): {class_uncert:.4f}")
#         # print("Class uncertainties (var):", np.round(class_uncert, 6))

#         pred_uncert = float(class_uncert[predicted_class])
#         print(f"Predicted class uncertainty (var): {pred_uncert:.4f}")



#     image_rgb, image_rgb_resized = preprocess_image_rgb(image_path)
#     scaled_contours, contour_overlay = scaling_contour(image_rgb, seg_map)
#     cv2.imwrite(os.path.join(contour_dir, filename), contour_overlay)

#     mask_overlay, overlay, binary_mask = scaled_overlay(image_rgb, scaled_contours)
#     cv2.imwrite(os.path.join(mask_overlay_dir, filename), mask_overlay)

#     drawing, shape, endpoints_sorted, skeleton_overlay = extremafit_2pts(image_rgb, scaled_contours, binary_mask)
#     print(f"Endpoints : {endpoints_sorted}")
#     cv2.imwrite(os.path.join(drawing_dir, filename), drawing)
#     cv2.imwrite(os.path.join(shape_dir, filename), shape)
#     cv2.imwrite(os.path.join(skeleton_overlay_dir, filename), skeleton_overlay)

# if __name__ == "__main__":
#     input_folder = "/home/htic/vasanth/bio_seg/data_parsed/bpd"
#     output_base_path = "/home/htic/vasanth/bio_seg/outputs/fetalnet/bpd_montecarlo"
#     os.makedirs(output_base_path, exist_ok=True)

#     contour_dir = os.path.join(output_base_path, "contour_overlay")
#     mask_overlay_dir = os.path.join(output_base_path, "mask_overlay")
#     drawing_dir = os.path.join(output_base_path, "drawing")
#     shape_dir = os.path.join(output_base_path, "shape")
#     skeleton_overlay_dir = os.path.join(output_base_path, "skeleton")
#     all_masks_dir = os.path.join(output_base_path, "dropout_masks")

#     for path in [contour_dir, mask_overlay_dir, drawing_dir, shape_dir, skeleton_overlay_dir, all_masks_dir]:
#         os.makedirs(path, exist_ok=True)

#     valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
#     image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
#                    if f.lower().endswith(valid_exts)]
#     image_paths.sort()

#     for img_path in tqdm(image_paths, desc="Processing images"):
#         process_image(img_path, contour_dir, mask_overlay_dir, drawing_dir, shape_dir, skeleton_overlay_dir, all_masks_dir)

###############################################################

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
    image_rgb = cv2.imread(img_path)
    image_rgb_resized = cv2.resize(image_rgb, target_size)
    return image_rgb, image_rgb_resized


# ---------------- INFERENCE ----------------
def run_inference(image_path, monte_carlo=False, n_samples=10, save_all_masks_dir=None, filename_prefix=""):
    """
    Run inference with optional Monte Carlo Dropout.
    Saves each sampled segmentation mask if save_all_masks_dir is provided.
    """
    input_tensor, resized_gray = preprocess_image_for_model(image_path)

    if monte_carlo:
        model.train()         # <-- critical: activate dropout
        enable_dropout(model) # <-- ensure dropout layers stay active

        class_outputs = []
        seg_outputs = []

        with torch.no_grad():
            for i in range(n_samples):
                class_output, seg_output = model(input_tensor)
                class_outputs.append(torch.softmax(class_output, dim=1).cpu().numpy())
                seg_outputs.append(seg_output.cpu().numpy())

                # Save each dropout mask
                if save_all_masks_dir is not None:
                    seg_map_i = seg_output.cpu().numpy().squeeze()
                    seg_map_uint8 = (np.clip(seg_map_i, 0, 1) * 255).astype(np.uint8)
                    out_path = os.path.join(save_all_masks_dir, f"{filename_prefix}_mask_sample_{i+1}.png")
                    cv2.imwrite(out_path, seg_map_uint8)

        # Compute averages and variances
        class_outputs = np.stack(class_outputs, axis=0)
        seg_outputs = np.stack(seg_outputs, axis=0)

        mean_class_probs = np.mean(class_outputs, axis=0).squeeze()
        mean_seg_map = np.mean(seg_outputs, axis=0).squeeze()

        predicted_class = int(np.argmax(mean_class_probs))
        class_uncertainty = np.var(class_outputs, axis=0).squeeze()
        seg_uncertainty = np.var(seg_outputs, axis=0).squeeze()

        # restore eval mode
        model.eval()
        return predicted_class, mean_class_probs, mean_seg_map, resized_gray, class_uncertainty, seg_uncertainty

    else:
        model.eval()
        with torch.no_grad():
            class_output, seg_output = model(input_tensor)
        class_probs = torch.softmax(class_output, dim=1).cpu().numpy().squeeze()
        predicted_class = int(np.argmax(class_probs))
        seg_map = seg_output.cpu().numpy().squeeze()
        return predicted_class, class_probs, seg_map, resized_gray, None, None


# ---------------- POSTPROCESS & VISUALIZATION ----------------
def scaling_contour(rgb_image, seg_map):
    img_h, img_w = rgb_image.shape[:2]
    mask = (seg_map * 255).astype(np.uint8)
    mask_h, mask_w = mask.shape[:2]

    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scale_x = img_w / mask_w
    scale_y = img_h / mask_h

    scaled_contours = []
    for cnt in contours:
        scaled_cnt = cnt.astype(np.float32)
        scaled_cnt[:, 0, 0] *= scale_x
        scaled_cnt[:, 0, 1] *= scale_y
        scaled_contours.append(scaled_cnt.astype(np.int32))

    img_with_contour = rgb_image.copy()
    cv2.drawContours(img_with_contour, scaled_contours, -1, (0, 255, 0), 5)
    contour_overlay = cv2.addWeighted(rgb_image, 0.5, img_with_contour, 0.8, 0)
    return scaled_contours, contour_overlay


def scaled_overlay(rgb_image, scaled_contours):
    overlay = np.zeros_like(rgb_image, dtype=np.uint8)
    cv2.fillPoly(overlay, scaled_contours, color=(0, 0, 255))
    binary_mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(binary_mask, scaled_contours, color=255)
    mask_overlay = cv2.addWeighted(rgb_image, 1.0, overlay, 0.8, 0)
    return mask_overlay, overlay, binary_mask


def extremafit_2pts(rgb_image, contours, binary_mask):
    drawing = rgb_image.copy()
    height, width = rgb_image.shape[:2]
    line_mask = np.zeros((height, width, 3), dtype=np.uint8)
    skeleton = skeletonize(binary_mask)
    skeleton_uint8 = (skeleton.astype(np.uint8) * 255)
    skeleton_color = np.zeros_like(rgb_image)
    skeleton_color[skeleton_uint8 > 0] = (0, 0, 255)
    skeleton_overlay = cv2.addWeighted(rgb_image, 0.7, skeleton_color, 0.3, 0)

    coords = cv2.findNonZero(skeleton_uint8)
    if coords is None:
        return drawing, line_mask, [], skeleton_overlay

    leftmost = tuple(coords[coords[:, :, 0].argmin()][0])
    rightmost = tuple(coords[coords[:, :, 0].argmax()][0])
    endpoints_sorted = [rightmost, leftmost]

    cv2.line(line_mask, leftmost, rightmost, color=(0, 255, 0), thickness=2)
    for pt in [leftmost, rightmost]:
        cv2.drawMarker(drawing, pt, (0, 255, 255), markerType=cv2.MARKER_CROSS,
                       markerSize=30, thickness=5, line_type=cv2.LINE_AA)
        cv2.circle(skeleton_overlay, pt, 5, (0, 255, 0), -1)
    return drawing, line_mask, endpoints_sorted, skeleton_overlay


# ---------------- PROCESS IMAGE ----------------
def process_image(image_path, contour_dir, mask_overlay_dir, drawing_dir, shape_dir, skeleton_overlay_dir, all_masks_dir):
    filename = os.path.basename(image_path)
    filename_wo_ext = os.path.splitext(filename)[0]

    predicted_class, class_probs, seg_map, resized_gray, class_uncert, seg_uncert = run_inference(
        image_path, monte_carlo=True, n_samples=10, save_all_masks_dir=all_masks_dir, filename_prefix=filename_wo_ext
    )

    print(f"\nProcessing: {image_path}")
    print("Predicted class:", predicted_class)


    if class_uncert is not None:
        print("Class variances (per class):", np.round(class_uncert, 6))
        pred_uncert = float(class_uncert[predicted_class])
        print(f"Predicted class uncertainty (var): {pred_uncert:.6f}")

    if seg_uncert is not None:
        print(f"Shape : {seg_uncert.shape}")
        print("Segmentation variances (per class):", np.round(seg_uncert, 6))
        mean_seg_uncert = float(np.mean(seg_uncert))
        print(f"Segmentation uncertainty (mean var): {mean_seg_uncert:.6f}")
    

    image_rgb, image_rgb_resized = preprocess_image_rgb(image_path)
    scaled_contours, contour_overlay = scaling_contour(image_rgb, seg_map)
    cv2.imwrite(os.path.join(contour_dir, filename), contour_overlay)

    mask_overlay, overlay, binary_mask = scaled_overlay(image_rgb, scaled_contours)
    cv2.imwrite(os.path.join(mask_overlay_dir, filename), mask_overlay)

    drawing, shape, endpoints_sorted, skeleton_overlay = extremafit_2pts(image_rgb, scaled_contours, binary_mask)
    print(f"Endpoints: {endpoints_sorted}")
    cv2.imwrite(os.path.join(drawing_dir, filename), drawing)
    cv2.imwrite(os.path.join(shape_dir, filename), shape)
    cv2.imwrite(os.path.join(skeleton_overlay_dir, filename), skeleton_overlay)


# ---------------- MAIN ----------------
if __name__ == "__main__":
    input_folder = "/home/htic/vasanth/bio_seg/data_parsed/bpd"
    output_base_path = "/home/htic/vasanth/bio_seg/outputs/fetalnet/bpd_montecarlo"
    os.makedirs(output_base_path, exist_ok=True)

    contour_dir = os.path.join(output_base_path, "contour_overlay")
    mask_overlay_dir = os.path.join(output_base_path, "mask_overlay")
    drawing_dir = os.path.join(output_base_path, "drawing")
    shape_dir = os.path.join(output_base_path, "shape")
    skeleton_overlay_dir = os.path.join(output_base_path, "skeleton")
    all_masks_dir = os.path.join(output_base_path, "dropout_masks")

    for path in [contour_dir, mask_overlay_dir, drawing_dir, shape_dir, skeleton_overlay_dir, all_masks_dir]:
        os.makedirs(path, exist_ok=True)

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
                   if f.lower().endswith(valid_exts)]
    image_paths.sort()

    for img_path in tqdm(image_paths, desc="Processing images"):
        process_image(img_path, contour_dir, mask_overlay_dir, drawing_dir, shape_dir, skeleton_overlay_dir, all_masks_dir)
