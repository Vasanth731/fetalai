import torch
import cv2
import numpy as np
import os
from tqdm import tqdm
from skimage.morphology import skeletonize
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import csv
from scipy.special import gamma
import warnings
warnings.filterwarnings('ignore')

################################# model for no marking images #############################################
class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=device),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=device))


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])

                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output = layer_output.permute(1, 0, 2, 3, 4)

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


def double_conv(input_channels, output_channels):
    """
    Args:
        input_channels:
        output_channels:
    Returns:
    """
    conv = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=output_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=output_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=0.2)
    )
    return conv


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')

    return src


class YNet(nn.Module):
    """
    UNet class.
    """

    def __init__(self, input_channels, output_channels, n_class):
        """
        Args:
            input_channels:
            output_channels:
            n_class:
        """
        super(YNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = double_conv(input_channels, output_channels)
        self.down_conv2 = double_conv(output_channels, 2 * output_channels)
        self.down_conv3 = double_conv(2 * output_channels, 4 * output_channels)
        self.down_conv4 = double_conv(4 * output_channels, 8 * output_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Linear(in_features=8 * output_channels, out_features=4)
        )

        self.bootleneck = double_conv(8 * output_channels, 16 * output_channels)

        self.up_transpose4 = nn.ConvTranspose2d(16 * output_channels, 8 * output_channels, kernel_size=2, stride=2)
        self.Att4 = Attention_block(F_g=8 * output_channels, F_l=8 * output_channels, F_int=4 * output_channels)
        self.up_conv4 = double_conv(16 * output_channels, 8 * output_channels)

        self.up_transpose3 = nn.ConvTranspose2d(8 * output_channels, 4 * output_channels, kernel_size=2, stride=2)
        self.Att3 = Attention_block(F_g=4 * output_channels, F_l=4 * output_channels, F_int=2 * output_channels)
        self.up_conv3 = double_conv(8 * output_channels, 4 * output_channels)

        self.up_transpose2 = nn.ConvTranspose2d(4 * output_channels, 2 * output_channels, kernel_size=2, stride=2)
        self.Att2 = Attention_block(F_g=2 * output_channels, F_l=2 * output_channels, F_int=1 * output_channels)
        self.up_conv2 = double_conv(4 * output_channels, 2 * output_channels)

        self.up_transpose1 = nn.ConvTranspose2d(2 * output_channels, output_channels, kernel_size=2, stride=2)
        self.Att1 = Attention_block(F_g=1 * output_channels, F_l=1 * output_channels, F_int=output_channels // 2)
        self.up_conv1 = double_conv(2 * output_channels, output_channels)
        self.out_layer = nn.Conv2d(4, out_channels=n_class, kernel_size=3, padding=1)
        self.conv_lstm = ConvLSTM(input_size=(14, 14),
                                  input_dim=512,
                                  hidden_dim=[512, 512],
                                  kernel_size=(3, 3),
                                  num_layers=2,
                                  batch_first=False,
                                  bias=True,
                                  return_all_layers=False)
        self.side1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        self.side2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1)
        self.side3 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1)
        self.side4 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Args:
            image:
        Returns:
        """
        # Encoder part
        x = torch.unbind(x, dim=1)
        data = []
        for item in x:
            x1 = self.down_conv1(item)  # [1, 64, 224, 224]
            x2 = self.down_conv2(self.max_pool_2x2(x1))  # # [1, 128, 112, 112]
            x3 = self.down_conv3(self.max_pool_2x2(x2))  # [1, 256, 56, 56]
            x4 = self.down_conv4(self.max_pool_2x2(x3))  # [1, 512, 28, 28]
            features = self.max_pool_2x2(x4)  # [1, 512, 14, 14]
            data.append(features.unsqueeze(0))
        data = torch.cat(data, dim=0)  # [1, 1, 512, 14, 14]
        lstm, _ = self.conv_lstm(data)
        test = lstm[0][-1, :, :, :, :]  # [1, 512, 14, 14]
        center = self.bootleneck(test)  # [1, 1024, 14, 14]

        # Classification
        avg_pool = self.avgpool(test)
        flatten = torch.flatten(avg_pool, 1)
        fc = self.classifier(flatten)

        # Decoder part
        up_transpose4 = self.up_transpose4(center)
        att4 = self.Att4(g=up_transpose4, x=x4)
        up_conv4 = self.up_conv4(torch.cat([x4, att4], dim=1))  # [1, 512, 28, 28]

        up_transpose3 = self.up_transpose3(up_conv4)
        att3 = self.Att3(g=up_transpose3, x=x3)
        up_conv3 = self.up_conv3(torch.cat([x3, att3], dim=1))  # [1, 256, 56, 56]

        up_transpose2 = self.up_transpose2(up_conv3)
        att2 = self.Att2(g=up_transpose2, x=x2)
        up_conv2 = self.up_conv2(torch.cat([x2, att2], dim=1))  # [1, 128, 112, 112]

        up_transpose1 = self.up_transpose1(up_conv2)
        att1 = self.Att1(g=up_transpose1, x=x1)
        up_conv1 = self.up_conv1(torch.cat([x1, att1], dim=1))  # [1, 64, 224, 224]

        # Side output
        s_1 = self.side1(up_conv1)  # [1, 1, 224, 224]
        s_2 = self.side2(up_conv2)
        s_2 = _upsample_like(s_2, s_1)
        s_3 = self.side3(up_conv3)
        s_3 = _upsample_like(s_3, s_1)
        s_4 = self.side4(up_conv4)
        s_4 = _upsample_like(s_4, s_1)
        out = self.out_layer(torch.cat((s_1, s_2, s_3, s_4), 1))  # [1, 1, 224, 224]
        out = torch.sigmoid(out)

        return fc, out


################################### inference for no marking images ##########################################
# ---------------- CONFIG ----------------
input_channels = 1
output_channels = 64
n_class = 1
target_size = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
model = YNet(input_channels=input_channels, output_channels=output_channels, n_class=n_class)
model.load_state_dict(torch.load("/home/htic/vasanth/fetalai_project/weights/fetalnet/pretrained/fuvai_weights.pt", map_location=device))
model = model.to(device)
model.eval()  # start in eval mode


# ---------------- ENABLE DROPOUT ----------------
def enable_dropout(model):
    """Keep dropout layers active during inference."""
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
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
def run_inference(image_path, monte_carlo=False, n_samples=10):
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
    ellipse_overlay = rgb_image.copy()  # Overlay for ellipse and axes on original image

    rows = []
    ellipse = None
    endpoints_sorted = []

    # Find the largest contour
    if len(contours) == 0:
        return drawing, ellipse_shape, ellipse, endpoints_sorted, ellipse_overlay

    largest_contour = max(contours, key=cv2.contourArea)
    contours = [largest_contour]  # Process only the largest contour

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

            # Draw ellipse on overlay
            cv2.ellipse(ellipse_overlay, ellipse, (0, 255, 0), 2)

            # Draw major and minor axis lines on overlay
            cv2.line(ellipse_overlay, (int(pt1_major[0]), int(pt1_major[1])),
                    (int(pt2_major[0]), int(pt2_major[1])), (255, 0, 0), 2)  # Blue for major axis
            cv2.line(ellipse_overlay, (int(pt1_minor[0]), int(pt1_minor[1])),
                    (int(pt2_minor[0]), int(pt2_minor[1])), (0, 0, 255), 2)  # Red for minor axis

            # points on the ellipse - draw only the 4 endpoint markers
            for pt in endpoints_sorted:
                x, y = int(pt[0]), int(pt[1])
                cv2.drawMarker(drawing, (x, y), (0, 255, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=25, thickness=3, line_type=cv2.LINE_AA)
                # cv2.circle(drawing, (x, y), 5, (0, 0, 255), -1)        # for the 720 image
                cv2.circle(ellipse_shape, (x, y), 5, (0, 0, 255), -1)  # for the later purpose to blow and overlay on the real image resolution

            # Ellipse and axis lines removed - only endpoint markers are drawn

    return drawing, ellipse_shape, ellipse, endpoints_sorted, ellipse_overlay


def extremafit_2pts(rgb_image, contours, binary_mask):
    drawing = rgb_image.copy()

    # Find the largest contour
    if len(contours) == 0:
        return drawing, np.zeros_like(rgb_image), [], rgb_image.copy()

    largest_contour = max(contours, key=cv2.contourArea)

    # Create binary mask from only the largest contour
    largest_binary_mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(largest_binary_mask, [largest_contour], 255)

    skeleton = skeletonize(largest_binary_mask)
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


def draw_arrowhead_markers(rgb_image, binary_mask):
    """Draw arrowheads pointing to the extrema points on the segmentation mask."""
    drawing = rgb_image.copy()

    # Find skeleton
    skeleton = skeletonize(binary_mask)
    skeleton_uint8 = (skeleton.astype(np.uint8) * 255)

    coords = cv2.findNonZero(skeleton_uint8)
    if coords is None:
        return drawing

    # Find extrema points
    leftmost = tuple(coords[coords[:,:,0].argmin()][0])
    rightmost = tuple(coords[coords[:,:,0].argmax()][0])
    topmost = tuple(coords[coords[:,:,1].argmin()][0])
    bottommost = tuple(coords[coords[:,:,1].argmax()][0])

    # Calculate center of mass for arrow origins
    M = cv2.moments(binary_mask)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        h, w = binary_mask.shape
        cx, cy = w // 2, h // 2

    # Draw arrows from center pointing to extrema
    arrow_color = (0, 255, 0)  # Green arrows
    thickness = 3
    tip_length = 0.3

    cv2.arrowedLine(drawing, (cx, cy), rightmost, arrow_color, thickness, tipLength=tip_length)
    cv2.arrowedLine(drawing, (cx, cy), leftmost, arrow_color, thickness, tipLength=tip_length)
    cv2.arrowedLine(drawing, (cx, cy), topmost, arrow_color, thickness, tipLength=tip_length)
    cv2.arrowedLine(drawing, (cx, cy), bottommost, arrow_color, thickness, tipLength=tip_length)

    # Draw center point
    cv2.circle(drawing, (cx, cy), 5, (255, 0, 0), -1)

    return drawing


def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) between two binary masks.

    Args:
        mask1: First binary mask (numpy array)
        mask2: Second binary mask (numpy array)

    Returns:
        float: IoU score between 0 and 1
    """
    # Ensure masks are binary
    mask1_binary = (mask1 > 0).astype(np.uint8)
    mask2_binary = (mask2 > 0).astype(np.uint8)

    # Calculate intersection and union
    intersection = np.logical_and(mask1_binary, mask2_binary).sum()
    union = np.logical_or(mask1_binary, mask2_binary).sum()

    # Avoid division by zero
    if union == 0:
        return 0.0

    iou = intersection / union
    return iou


def calculate_dice_coefficient(mask1, mask2):
    """
    Calculate Dice coefficient between two binary masks.

    Args:
        mask1: First binary mask (numpy array)
        mask2: Second binary mask (numpy array)

    Returns:
        float: Dice coefficient between 0 and 1
    """
    # Ensure masks are binary
    mask1_binary = (mask1 > 0).astype(np.uint8)
    mask2_binary = (mask2 > 0).astype(np.uint8)

    # Calculate intersection
    intersection = np.logical_and(mask1_binary, mask2_binary).sum()

    # Calculate Dice coefficient
    dice = (2.0 * intersection) / (mask1_binary.sum() + mask2_binary.sum())

    return dice


def visualize_overlap(image, mask1, mask2, output_path):
    """
    Visualize the overlap between two masks.
    - Green: mask1 only
    - Red: mask2 only
    - Yellow: overlap between mask1 and mask2

    Args:
        image: Original image
        mask1: First binary mask (ellipse from arrowhead detection)
        mask2: Second binary mask (segmentation mask)
        output_path: Path to save the visualization
    """
    # Create a color overlay
    overlay = np.zeros_like(image)

    # Ensure masks are binary
    mask1_binary = (mask1 > 0).astype(np.uint8)
    mask2_binary = (mask2 > 0).astype(np.uint8)

    # Compute overlap
    overlap = np.logical_and(mask1_binary, mask2_binary)
    mask1_only = np.logical_and(mask1_binary, np.logical_not(mask2_binary))
    mask2_only = np.logical_and(mask2_binary, np.logical_not(mask1_binary))

    # Apply colors
    overlay[mask1_only] = [0, 255, 0]  # Green for ellipse only
    overlay[mask2_only] = [0, 0, 255]  # Red for segmentation only
    overlay[overlap] = [0, 255, 255]   # Yellow for overlap

    # Blend with original image
    result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

    # Save the result
    cv2.imwrite(output_path, result)
    print(f"Overlap visualization saved to: {output_path}")

    return result


# ---------------- PROCESS IMAGE ----------------
def process_image(image_path, contour_dir, mask_overlay_dir, drawing_dir, skeleton_overlay_dir, colormap_overlay_dir, ellipse_axis_overlay_dir, measurement_type):
    filename = os.path.basename(image_path)
    filename_noext = os.path.splitext(filename)[0]
    file_ext = os.path.splitext(filename)[1]

    predicted_class, class_probs, seg_map, resized_gray, class_uncert, seg_uncert = run_inference(
        image_path, monte_carlo=True, n_samples=10
    )

    # Calculate segmentation metrics
    seg_confidence = float(seg_map[seg_map >= 0.1].mean()) if np.sum(seg_map >= 0.1) > 0 else 0.0
    seg_uncertainty_mean = float(np.mean(seg_uncert)) if seg_uncert is not None else 0.0

    # Get classification metrics
    class_confidence = float(class_probs[predicted_class])
    class_uncertainty_val = float(class_uncert[predicted_class]) if class_uncert is not None else 0.0

    # Print all metrics
    print(f"\n{'='*60}")
    print(f"Image: {filename}")
    print(f"{'='*60}")
    print(f"CLASSIFICATION:")
    print(f"  Predicted Class: {predicted_class}")
    print(f"  Class Confidence: {class_confidence:.4f}")
    print(f"  Class Uncertainty: {class_uncertainty_val:.6f}")
    print(f"\nSEGMENTATION:")
    print(f"  Segmentation Confidence: {seg_confidence:.4f}")
    print(f"  Segmentation Uncertainty: {seg_uncertainty_mean:.6f}")
    print(f"{'='*60}\n")

    # Create filename with segmentation scores
    scored_filename = f"{filename_noext}_segconf{seg_confidence:.4f}_segunc{seg_uncertainty_mean:.6f}{file_ext}"

    # Read original image for overlays
    image_rgb, _ = preprocess_image_rgb(image_path)
    OG_image = cv2.imread(image_path)
    OG_Shape = OG_image.shape

    # Create colormap overlay like infer_score2.py
    seg_output_viz = (np.expand_dims(seg_map, -1) * 255).astype(np.uint8)
    seg_output_viz = np.stack((seg_output_viz.squeeze(),) * 3, axis=-1)
    mask_color = cv2.resize(seg_output_viz, (OG_Shape[1], OG_Shape[0]))
    mask_color = cv2.applyColorMap(mask_color, cv2.COLORMAP_JET)
    colormap_overlay = cv2.addWeighted(OG_image, 0.7, mask_color, 0.3, 0)
    cv2.imwrite(os.path.join(colormap_overlay_dir, scored_filename), colormap_overlay)

    scaled_contours, contour_overlay = scaling_contour(image_rgb, seg_map)
    cv2.imwrite(os.path.join(contour_dir, scored_filename), contour_overlay)

    mask_overlay, overlay, binary_mask = scaled_overlay(image_rgb, scaled_contours)
    cv2.imwrite(os.path.join(mask_overlay_dir, scored_filename), mask_overlay)

    # Use different fitting functions based on measurement type
    if measurement_type in ["02_bpd_hc_transthalamic", "01_abdominal_circumference"]:
        drawing, shape, ellipse, endpoints_sorted, ellipse_overlay = boundingbox_ellipsefit(image_rgb, scaled_contours)
        skeleton_overlay = np.zeros_like(image_rgb)  # No skeleton overlay for ellipse fit
        # Save ellipse overlay with major and minor axis lines
        cv2.imwrite(os.path.join(ellipse_axis_overlay_dir, scored_filename), ellipse_overlay)
    elif measurement_type == "03_femur_length":
        drawing, shape, endpoints_sorted, skeleton_overlay = extremafit_2pts(image_rgb, scaled_contours, binary_mask)
    else:
        print(f"Warning: Unknown measurement type '{measurement_type}', using extremafit_2pts as default")
        drawing, shape, endpoints_sorted, skeleton_overlay = extremafit_2pts(image_rgb, scaled_contours, binary_mask)

    print(f"Endpoints: {endpoints_sorted}")
    cv2.imwrite(os.path.join(drawing_dir, scored_filename), drawing)
    cv2.imwrite(os.path.join(skeleton_overlay_dir, scored_filename), skeleton_overlay)

    # Resize binary_mask to original image size for comparison
    binary_mask_full_size = cv2.resize(binary_mask, (OG_Shape[1], OG_Shape[0]), interpolation=cv2.INTER_NEAREST)

    return binary_mask_full_size, OG_image, seg_confidence, seg_uncertainty_mean



####################################### model and inference for marked images ####################################
def detect_arrowhead(path, net, conf=0.5):

    # determine the output layers
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    img0 = cv2.imread(path)
    img = img0.copy()

    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    t0 = time.time()
    outputs = net.forward(ln)
    t = time.time() - t0

    # combine the 3 output groups into 1 (10647, 85)
    # large objects (507, 85)
    # medium objects (2028, 85)
    # small objects (8112, 85)
    # outputs = outputs[-1]
    outputs = np.vstack(outputs)

    H, W = img.shape[:2]

    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > conf:
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            p0 = int(x - w//2), int(y - h//2)
            p1 = int(x + w//2), int(y + h//2)
            boxes.append([*p0, int(w), int(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
    locs = []
    if len(indices)>0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            center = int(x + w//2), int(y + h//2)
            locs.append(center)

            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
            text = "{}: {:.4f}".format("ArrowHead", confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    else:
        locs = None
    return img, locs


def detect_all(data_root, yolo_weight, yolo_cfg, num_points, dest="out_yolo", save_figs=False, single_image=False):
    """
    Detect arrowheads using YOLO on either a single image or a folder of images.

    Args:
        data_root: Path to a folder containing images OR path to a single image
        yolo_weight: Path to YOLO weights file
        yolo_cfg: Path to YOLO config file
        num_points: Expected number of arrowhead points
        dest: Output directory for saving results
        save_figs: Whether to save detection visualizations
        single_image: If True, data_root is treated as a single image path

    Returns:
        For single_image=True: tuple of (annotated_img, locs, filename)
        For single_image=False: dictionary of {filename: locs}
    """
    # Check if input is a single image
    if single_image or os.path.isfile(data_root):
        single_image = True
        imgs = [os.path.basename(data_root)]
        img_dir = os.path.dirname(data_root)
        if img_dir == "":
            img_dir = "."
    else:
        imgs = os.listdir(data_root)
        img_dir = data_root

    if save_figs:
        os.makedirs(dest, exist_ok=True)
        os.makedirs(os.path.join(dest, "bad"), exist_ok=True)
        os.makedirs(os.path.join(dest, "good"), exist_ok=True)

    # Give the configuration and weight files for the model and load the network.
    net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weight)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    data = {}
    for x in imgs:
        path = os.path.join(img_dir, x)
        img, locs = detect_arrowhead(path, net)

        if save_figs:
            if locs == None or len(locs) != num_points:
                cv2.imwrite(os.path.join(dest, "bad", x), img)
            else:
                cv2.imwrite(os.path.join(dest, "good", x), img)

        if locs is not None and len(locs)==num_points:
            data[x] = locs

    # Return different formats based on single_image mode
    if single_image:
        filename = imgs[0]
        if filename in data:
            return img, data[filename], filename
        else:
            return img, None, filename
    else:
        return data


def save_detections_to_csv(data, csv_filename="detections.csv"):
    """Saves detected bounding box coordinates along with filenames to a CSV file."""
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Coordinates"])

        for filename, coordinates in data.items():
            writer.writerow([filename, coordinates])


#################################### evaluation metrics for imagw quality ##################################

def calculate_brisque(img):
    """
    Calculate BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)

    Args:
        img: Input image

    Returns:
        float: BRISQUE score in range [0, 100], where lower is better

    Quality Interpretation:
        0-20  : Excellent quality
        20-40 : Good quality
        40-60 : Fair quality
        60-80 : Poor quality
        80-100: Very poor quality
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray = gray.astype(np.float64)

    # Normalize to zero mean and unit variance
    def compute_features(img):
        # MSCN (Mean Subtracted Contrast Normalized) coefficients
        mu = cv2.GaussianBlur(img, (7, 7), 7/6)
        mu_sq = mu * mu
        sigma = cv2.GaussianBlur(img * img, (7, 7), 7/6)
        sigma = np.sqrt(np.abs(sigma - mu_sq))

        # Avoid division by zero
        mscn = (img - mu) / (sigma + 1)

        # Calculate features from MSCN coefficients
        alpha, beta_l, beta_r = estimate_aggd_params(mscn)

        return [alpha, (beta_l + beta_r) / 2]

    def estimate_aggd_params(vec):
        # Estimate parameters of Asymmetric Generalized Gaussian Distribution
        vec = vec.flatten()

        # Calculate gamma parameters
        gam = np.arange(0.2, 10, 0.001)
        r_gam = (gamma(2/gam) ** 2) / (gamma(1/gam) * gamma(3/gam))

        # Calculate left and right variance
        left_std = np.sqrt(np.mean((vec[vec < 0]) ** 2))
        right_std = np.sqrt(np.mean((vec[vec > 0]) ** 2))

        # Calculate gamma mean
        gamma_mean = np.sqrt(np.mean(vec ** 2))

        # Prevent division by zero
        if gamma_mean < 1e-10:
            gamma_mean = 1e-10

        # Calculate r
        r = np.mean(np.abs(vec)) ** 2 / np.mean(vec ** 2)
        r = np.clip(r, 0.2, 0.999)

        # Find best matching gamma
        array_position = np.argmin(np.abs(r_gam - r))
        alpha = gam[array_position]

        return alpha, left_std, right_std

    # Extract features
    features = compute_features(gray)

    # Simplified BRISQUE score (actual BRISQUE uses SVR on extracted features)
    # This is a simplified version that gives an approximation
    # Lower alpha and higher beta indicate better quality
    score = abs(features[0] - 1.0) * 50 + (1.0 / (features[1] + 0.1)) * 10
    score = np.clip(score, 0, 100)

    return score


def calculate_niqe(img):
    """
    Calculate NIQE (Natural Image Quality Evaluator)

    Args:
        img: Input image

    Returns:
        float: NIQE score in range [0, ∞], where lower is better

    Quality Interpretation:
        0-3 : Excellent quality
        3-5 : Good quality
        5-7 : Fair quality
        7-10: Poor quality
        > 10: Very poor quality
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray = gray.astype(np.float64)

    # Patch size
    patch_size = 96

    # Calculate MSCN coefficients
    mu = cv2.GaussianBlur(gray, (7, 7), 7/6)
    mu_sq = mu * mu
    sigma = cv2.GaussianBlur(gray * gray, (7, 7), 7/6)
    sigma = np.sqrt(np.abs(sigma - mu_sq))

    # MSCN coefficients
    mscn = (gray - mu) / (sigma + 1)

    # Extract patches
    h, w = mscn.shape
    num_patches = 0
    feature_vectors = []

    for i in range(0, h - patch_size, patch_size // 2):
        for j in range(0, w - patch_size, patch_size // 2):
            patch = mscn[i:i+patch_size, j:j+patch_size]

            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                # Extract features from patch
                patch_mean = np.mean(patch)
                patch_var = np.var(patch)

                feature_vectors.append([patch_mean, patch_var])
                num_patches += 1

    if num_patches == 0:
        return 10.0  # Return poor quality if no patches found

    # Convert to numpy array
    features = np.array(feature_vectors)

    # Calculate multivariate Gaussian parameters
    mu_features = np.mean(features, axis=0)

    # Natural scene statistics parameters (pre-trained on pristine images)
    # These are simplified approximations
    mu_pristine = np.array([0.0, 1.0])  # Mean of pristine images

    # Calculate NIQE score as distance from pristine statistics
    niqe_score = np.linalg.norm(mu_features - mu_pristine)

    return niqe_score


# ======================== HELPER FUNCTIONS ========================

def interpret_quality(metric_name, score):
    """
    Interpret the quality score and provide a description
    """

    if metric_name == "BRISQUE":
        if score < 20:
            return "Excellent quality"
        elif score < 40:
            return "Good quality"
        elif score < 60:
            return "Fair quality"
        elif score < 80:
            return "Poor quality"
        else:
            return "Very poor quality"

    elif metric_name == "NIQE":
        if score < 3:
            return "Excellent quality"
        elif score < 5:
            return "Good quality"
        elif score < 7:
            return "Fair quality"
        elif score < 10:
            return "Poor quality"
        else:
            return "Very poor quality"

    return "Unknown"


def print_results(metric_name, score, score_range, interpretation):
    """
    Print formatted results
    """
    print(f"\n{'='*60}")
    print(f"Metric: {metric_name}")
    print(f"{'='*60}")
    print(f"Score: {score:.4f}")
    print(f"Range: {score_range}")
    print(f"Quality Assessment: {interpretation}")
    print(f"{'='*60}")



def assess_image_quality(test_img_path, reference_img_path=None):
    """
    Assess image quality using multiple metrics

    Args:
        test_img_path: Path to the test image
        reference_img_path: Path to the reference image (optional, for FR metrics)

    Returns:
        tuple: (brisque_score, niqe_score)
    """
    # Load test image
    test_img = cv2.imread(test_img_path)
    if test_img is None:
        print(f"Error: Could not load test image from {test_img_path}")
        return None, None

    print("\n" + "="*60)
    print("FETAL ULTRASOUND IMAGE QUALITY ASSESSMENT")
    print("="*60)
    print(f"Test Image: {test_img_path}")

    # No-Reference Metrics
    print("\n" + "="*60)
    print("NO-REFERENCE METRICS")
    print("="*60)

    # Calculate BRISQUE
    brisque_score = calculate_brisque(test_img)
    interpretation = interpret_quality("BRISQUE", brisque_score)
    print_results("BRISQUE (Blind Image Spatial Quality)",
                 brisque_score,
                 "[0, 100] (Lower is better)",
                 interpretation)

    # Calculate NIQE
    niqe_score = calculate_niqe(test_img)
    interpretation = interpret_quality("NIQE", niqe_score)
    print_results("NIQE (Natural Image Quality Evaluator)",
                 niqe_score,
                 "[0, ∞] (Lower is better)",
                 interpretation)

    print("\n" + "="*60)
    print("ASSESSMENT COMPLETE")
    print("="*60 + "\n")

    return brisque_score, niqe_score


def draw_ellipse_from_4_points(image, points, output_path):
    """
    Draw an ellipse from 4 detected arrowhead points with major and minor axes.
    Creates a semi-transparent overlay of the ellipse on the original image.

    Args:
        image: Original image (already loaded)
        points: List of 4 (x, y) tuples
        output_path: Path to save the output image

    Returns:
        tuple: (img_with_ellipse, ellipse_mask)
            - img_with_ellipse: Image with ellipse and axes drawn as overlay
            - ellipse_mask: Binary mask of the ellipse
    """
    # Create a copy of the original image
    img_with_ellipse = image.copy()

    # Create an overlay for the filled ellipse
    overlay = image.copy()

    # Convert points to numpy array
    points_array = np.array(points, dtype=np.float32)

    # To use cv2.fitEllipse, we need at least 5 points
    # Add a point that is 5 pixels away from the first known point as the 5th point
    # Offset by 5 pixels in the x-direction from the first point
    fifth_point = points_array[0] + np.array([5, 0], dtype=np.float32)
    points_with_center = np.vstack([points_array, fifth_point])

    # Fit ellipse to the 5 points (4 detected + fifth point offset by 5 pixels)
    # Reshape for cv2.fitEllipse format: (N, 1, 2)
    points_for_fitting = points_with_center.reshape(-1, 1, 2).astype(np.int32)
    ellipse = cv2.fitEllipse(points_for_fitting)

    (xc, yc), (d1, d2), angle = ellipse

    # Ensure major axis is larger
    if d1 < d2:
        d1, d2 = d2, d1
        angle = (angle + 90) % 180

    major_axis = d1
    minor_axis = d2

    print(f"Ellipse parameters - Center: ({xc:.2f}, {yc:.2f}), Major axis: {major_axis:.2f}, Minor axis: {minor_axis:.2f}, Angle: {angle:.2f}°")

    # Create a binary mask of the ellipse
    ellipse_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.ellipse(ellipse_mask, ellipse, 255, -1)  # Filled ellipse in white

    # Draw a filled semi-transparent ellipse on the overlay
    cv2.ellipse(overlay, ellipse, (0, 255, 0), -1)  # Filled ellipse in green

    # Blend the overlay with the original image (30% opacity for the filled ellipse)
    img_with_ellipse = cv2.addWeighted(overlay, 0.3, img_with_ellipse, 0.7, 0)

    # Draw the ellipse outline (solid line for clear boundary)
    cv2.ellipse(img_with_ellipse, ellipse, (0, 255, 0), 3)

    # Calculate endpoints of major and minor axes using the fitted ellipse parameters
    theta = np.radians(angle)

    # Major axis endpoints
    dx_major = (major_axis / 2) * np.cos(theta)
    dy_major = (major_axis / 2) * np.sin(theta)
    pt1_major = (int(xc + dx_major), int(yc + dy_major))
    pt2_major = (int(xc - dx_major), int(yc - dy_major))

    # Minor axis endpoints
    dx_minor = (minor_axis / 2) * -np.sin(theta)
    dy_minor = (minor_axis / 2) * np.cos(theta)
    pt1_minor = (int(xc + dx_minor), int(yc + dy_minor))
    pt2_minor = (int(xc - dx_minor), int(yc - dy_minor))

    # Draw major axis (blue) with thicker line
    cv2.line(img_with_ellipse, pt1_major, pt2_major, (255, 0, 0), 3)

    # Draw minor axis (red) with thicker line
    cv2.line(img_with_ellipse, pt1_minor, pt2_minor, (0, 0, 255), 3)

    # Draw center point
    cv2.circle(img_with_ellipse, (int(xc), int(yc)), 5, (255, 255, 0), -1)

    # Draw the detected arrowhead points
    for pt in points:
        cv2.circle(img_with_ellipse, (int(pt[0]), int(pt[1])), 8, (0, 255, 255), -1)

    # Save the image
    cv2.imwrite(output_path, img_with_ellipse)
    print(f"Ellipse overlay saved to: {output_path}")
    print(f"  - Major axis (blue line): {major_axis:.2f} pixels")
    print(f"  - Minor axis (red line): {minor_axis:.2f} pixels")

    return img_with_ellipse, ellipse_mask


def draw_line_from_2_points(image, points, output_path):
    """
    Draw a straight line between 2 detected arrowhead points.

    Args:
        image: Original image (already loaded)
        points: List of 2 (x, y) tuples
        output_path: Path to save the output image

    Returns:
        Image with line drawn
    """
    img_with_line = image.copy()

    # Draw line between the two points
    pt1 = (int(points[0][0]), int(points[0][1]))
    pt2 = (int(points[1][0]), int(points[1][1]))

    cv2.line(img_with_line, pt1, pt2, (0, 255, 0), 3)

    # Draw the detected points
    for pt in points:
        cv2.circle(img_with_line, (int(pt[0]), int(pt[1])), 8, (0, 255, 255), -1)

    # Save the image
    cv2.imwrite(output_path, img_with_line)
    print(f"Line overlay saved to: {output_path}")

    return img_with_line

def process_folder(input_folder, output_base, measurement_type, brisque_threshold=60, niqe_threshold=7,
                   seg_confidence_threshold=0.5, seg_uncertainty_threshold=0.1):
    """
    Process all images in a folder through the complete pipeline.

    Args:
        input_folder: Path to folder containing input images
        output_base: Base directory for all outputs
        measurement_type: Type of measurement ("02_bpd_hc_transthalamic", "01_abdominal_circumference", "03_femur_length")
        brisque_threshold: Maximum acceptable BRISQUE score (default: 60)
        niqe_threshold: Maximum acceptable NIQE score (default: 7)
        seg_confidence_threshold: Minimum segmentation confidence (default: 0.5)
        seg_uncertainty_threshold: Maximum segmentation uncertainty (default: 0.1)

    Returns:
        dict: Summary statistics of processing results
    """
    # YOLO configuration
    yolo_weight = "/home/htic/vasanth/fetalai_project/fetalai/yolo_arrowhead/yolov4-tiny-arrowhead.weights"
    yolo_cfg = "/home/htic/vasanth/fetalai_project/fetalai/yolo_arrowhead/yolov4-tiny-arrowhead.cfg"

    # Create main output directory
    os.makedirs(output_base, exist_ok=True)

    # Get all image files from input folder
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for file in os.listdir(input_folder):
        if any(file.lower().endswith(fmt) for fmt in supported_formats):
            image_files.append(os.path.join(input_folder, file))

    if len(image_files) == 0:
        print(f"No images found in {input_folder}")
        return {}

    print(f"\nFound {len(image_files)} images to process")
    print(f"Measurement type: {measurement_type}")
    print(f"BRISQUE Threshold: {brisque_threshold}")
    print(f"NIQE Threshold: {niqe_threshold}")
    print(f"Segmentation Confidence Threshold: {seg_confidence_threshold}")
    print(f"Segmentation Uncertainty Threshold: {seg_uncertainty_threshold}")
    print(f"Output directory: {output_base}\n")

    # Statistics tracking
    stats = {
        'total_images': len(image_files),
        'quality_pass': 0,
        'quality_fail': 0,
        'arrowheads_detected': 0,
        'no_arrowheads': 0,
        'ellipse_drawn': 0,
        'line_drawn': 0,
        'processing_errors': 0,
        'above_threshold': 0,
        'below_threshold': 0
    }

    # Process each image
    for idx, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            filename = os.path.basename(image_path)
            print(f"\n{'='*80}")
            print(f"Processing image {idx+1}/{len(image_files)}: {filename}")
            print(f"{'='*80}")

            # Step 1: Assess Image Quality
            print("\n" + "="*60)
            print("STEP 1: ASSESSING IMAGE QUALITY")
            print("="*60)
            brisque_score, niqe_score = assess_image_quality(image_path)

            if brisque_score is None or niqe_score is None:
                print(f"Error: Could not assess image quality for {filename}. Skipping.")
                stats['processing_errors'] += 1
                continue

            # Check quality thresholds
            quality_pass = True
            if brisque_score > brisque_threshold:
                print(f"\nWarning: BRISQUE score ({brisque_score:.4f}) exceeds threshold ({brisque_threshold})")
                quality_pass = False
            else:
                print(f"\nBRISQUE score ({brisque_score:.4f}) passes threshold ({brisque_threshold})")

            if niqe_score > niqe_threshold:
                print(f"Warning: NIQE score ({niqe_score:.4f}) exceeds threshold ({niqe_threshold})")
                quality_pass = False
            else:
                print(f"NIQE score ({niqe_score:.4f}) passes threshold ({niqe_threshold})")

            if quality_pass:
                stats['quality_pass'] += 1
                print("\nImage quality is acceptable. Proceeding with detection.")
            else:
                stats['quality_fail'] += 1
                print("\nImage quality is below acceptable thresholds.")
                print("Continuing with processing anyway...")

            # Step 2: Detect Arrowheads
            print("\n" + "="*60)
            print("STEP 2: DETECTING ARROWHEADS")
            print("="*60)

            img, locs, _ = detect_all(
                data_root=image_path,
                yolo_weight=yolo_weight,
                yolo_cfg=yolo_cfg,
                num_points=4,
                save_figs=True,
                dest=os.path.join(output_base, "yolo_detection"),
                single_image=True
            )

            if locs is None:
                stats['no_arrowheads'] += 1
                print("\nNo arrowheads detected!")
                print("="*60)
                print("STEP 3: RUNNING PROCESS_IMAGE FOR NO-MARKING SCENARIO")
                print("="*60)

                # Run process_image to get segmentation metrics
                # Create temporary output directories to get metrics first
                temp_contour_dir = os.path.join(output_base, "temp", "contour_overlay")
                temp_mask_overlay_dir = os.path.join(output_base, "temp", "mask_overlay")
                temp_drawing_dir = os.path.join(output_base, "temp", "drawing")
                temp_skeleton_overlay_dir = os.path.join(output_base, "temp", "skeleton")
                temp_colormap_overlay_dir = os.path.join(output_base, "temp", "colormap_overlay")
                temp_ellipse_axis_overlay_dir = os.path.join(output_base, "temp", "ellipse_axis_overlay")

                for p in [temp_contour_dir, temp_mask_overlay_dir, temp_drawing_dir, temp_skeleton_overlay_dir,
                          temp_colormap_overlay_dir, temp_ellipse_axis_overlay_dir]:
                    os.makedirs(p, exist_ok=True)

                # Process image without markers to get metrics
                _, _, seg_confidence, seg_uncertainty = process_image(
                    image_path, temp_contour_dir, temp_mask_overlay_dir, temp_drawing_dir,
                    temp_skeleton_overlay_dir, temp_colormap_overlay_dir, temp_ellipse_axis_overlay_dir,
                    measurement_type
                )

                # Determine if image meets threshold
                meets_threshold = (seg_confidence >= seg_confidence_threshold and
                                 seg_uncertainty <= seg_uncertainty_threshold)

                # Choose output folder based on threshold
                if meets_threshold:
                    threshold_folder = "above_threshold"
                    stats['above_threshold'] += 1
                    print(f"\nSegmentation meets threshold criteria:")
                    print(f"  Confidence: {seg_confidence:.4f} >= {seg_confidence_threshold}")
                    print(f"  Uncertainty: {seg_uncertainty:.4f} <= {seg_uncertainty_threshold}")
                else:
                    threshold_folder = "below_threshold"
                    stats['below_threshold'] += 1
                    print(f"\nSegmentation below threshold criteria:")
                    print(f"  Confidence: {seg_confidence:.4f} < {seg_confidence_threshold} OR")
                    print(f"  Uncertainty: {seg_uncertainty:.4f} > {seg_uncertainty_threshold}")

                # Create final output directories based on threshold
                contour_dir = os.path.join(output_base, threshold_folder, "contour_overlay")
                mask_overlay_dir = os.path.join(output_base, threshold_folder, "mask_overlay")
                drawing_dir = os.path.join(output_base, threshold_folder, "drawing")
                skeleton_overlay_dir = os.path.join(output_base, threshold_folder, "skeleton")
                colormap_overlay_dir = os.path.join(output_base, threshold_folder, "colormap_overlay")
                ellipse_axis_overlay_dir = os.path.join(output_base, threshold_folder, "ellipse_axis_overlay")

                for p in [contour_dir, mask_overlay_dir, drawing_dir, skeleton_overlay_dir,
                          colormap_overlay_dir, ellipse_axis_overlay_dir]:
                    os.makedirs(p, exist_ok=True)

                # Process image again to save to correct directories
                process_image(image_path, contour_dir, mask_overlay_dir, drawing_dir,
                             skeleton_overlay_dir, colormap_overlay_dir, ellipse_axis_overlay_dir,
                             measurement_type)

                # Clean up temp directory
                import shutil
                shutil.rmtree(os.path.join(output_base, "temp"), ignore_errors=True)

                print(f"\nNo-marking processing complete for {filename}")
                print(f"Saved to: {threshold_folder}/")

            else:
                stats['arrowheads_detected'] += 1
                print(f"\nArrowheads detected at locations: {locs}")

                # Step 3: Run process_image first to get segmentation metrics
                print("\n" + "="*60)
                print("STEP 3: RUNNING PROCESS_IMAGE FOR SEGMENTATION")
                print("="*60)

                # Create temporary output directories to get metrics first
                temp_contour_dir = os.path.join(output_base, "temp", "contour_overlay")
                temp_mask_overlay_dir = os.path.join(output_base, "temp", "mask_overlay")
                temp_drawing_dir = os.path.join(output_base, "temp", "drawing")
                temp_skeleton_overlay_dir = os.path.join(output_base, "temp", "skeleton")
                temp_colormap_overlay_dir = os.path.join(output_base, "temp", "colormap_overlay")
                temp_ellipse_axis_overlay_dir = os.path.join(output_base, "temp", "ellipse_axis_overlay")

                for p in [temp_contour_dir, temp_mask_overlay_dir, temp_drawing_dir, temp_skeleton_overlay_dir,
                          temp_colormap_overlay_dir, temp_ellipse_axis_overlay_dir]:
                    os.makedirs(p, exist_ok=True)

                # Process image to get segmentation mask and metrics
                seg_mask, seg_image, seg_confidence, seg_uncertainty = process_image(
                    image_path, temp_contour_dir, temp_mask_overlay_dir, temp_drawing_dir,
                    temp_skeleton_overlay_dir, temp_colormap_overlay_dir, temp_ellipse_axis_overlay_dir,
                    measurement_type
                )

                # Determine if image meets threshold
                meets_threshold = (seg_confidence >= seg_confidence_threshold and
                                 seg_uncertainty <= seg_uncertainty_threshold)

                # Choose output folder based on threshold
                if meets_threshold:
                    threshold_folder = "above_threshold"
                    stats['above_threshold'] += 1
                    print(f"\nSegmentation meets threshold criteria:")
                    print(f"  Confidence: {seg_confidence:.4f} >= {seg_confidence_threshold}")
                    print(f"  Uncertainty: {seg_uncertainty:.4f} <= {seg_uncertainty_threshold}")
                else:
                    threshold_folder = "below_threshold"
                    stats['below_threshold'] += 1
                    print(f"\nSegmentation below threshold criteria:")
                    print(f"  Confidence: {seg_confidence:.4f} < {seg_confidence_threshold} OR")
                    print(f"  Uncertainty: {seg_uncertainty:.4f} > {seg_uncertainty_threshold}")

                # Step 4: Draw ellipse or line based on number of points
                print("\n" + "="*60)
                print("STEP 4: DRAWING OVERLAY BASED ON DETECTED POINTS")
                print("="*60)

                # Create output directory for marked overlays based on threshold
                marked_overlay_dir = os.path.join(output_base, threshold_folder, "marked_overlay")
                os.makedirs(marked_overlay_dir, exist_ok=True)

                # Load the original image
                original_image = cv2.imread(image_path)

                if original_image is None:
                    print(f"Error: Could not load image from {image_path}")
                    stats['processing_errors'] += 1
                    continue

                output_filename = os.path.join(marked_overlay_dir, filename)

                if len(locs) == 4:
                    stats['ellipse_drawn'] += 1
                    print(f"Drawing ellipse with major and minor axes from 4 points...")
                    img_with_ellipse, ellipse_mask = draw_ellipse_from_4_points(original_image, locs, output_filename)

                    # Step 5: Create final output directories based on threshold
                    print("\n" + "="*60)
                    print("STEP 5: SAVING SEGMENTATION OUTPUTS TO THRESHOLD FOLDERS")
                    print("="*60)

                    # Create final output directories for process_image based on threshold
                    contour_dir = os.path.join(output_base, threshold_folder, "contour_overlay")
                    mask_overlay_dir = os.path.join(output_base, threshold_folder, "mask_overlay")
                    drawing_dir = os.path.join(output_base, threshold_folder, "drawing")
                    skeleton_overlay_dir = os.path.join(output_base, threshold_folder, "skeleton")
                    colormap_overlay_dir = os.path.join(output_base, threshold_folder, "colormap_overlay")
                    ellipse_axis_overlay_dir = os.path.join(output_base, threshold_folder, "ellipse_axis_overlay")

                    for p in [contour_dir, mask_overlay_dir, drawing_dir, skeleton_overlay_dir,
                              colormap_overlay_dir, ellipse_axis_overlay_dir]:
                        os.makedirs(p, exist_ok=True)

                    # Process image again to save to correct directories
                    process_image(image_path, contour_dir, mask_overlay_dir, drawing_dir,
                                 skeleton_overlay_dir, colormap_overlay_dir, ellipse_axis_overlay_dir,
                                 measurement_type)

                    # Step 6: Compare masks and visualize overlap
                    print("\n" + "="*60)
                    print("STEP 6: COMPARING ELLIPSE AND SEGMENTATION MASKS")
                    print("="*60)

                    # Calculate overlap metrics
                    iou_score = calculate_iou(ellipse_mask, seg_mask)
                    dice_score = calculate_dice_coefficient(ellipse_mask, seg_mask)

                    print(f"\nOverlap Metrics:")
                    print(f"  IoU (Intersection over Union): {iou_score:.4f}")
                    print(f"  Dice Coefficient: {dice_score:.4f}")
                    print(f"\n  Interpretation:")
                    if iou_score >= 0.7:
                        print(f"    - Excellent agreement between ellipse and segmentation (IoU >= 0.7)")
                    elif iou_score >= 0.5:
                        print(f"    - Good agreement between ellipse and segmentation (IoU >= 0.5)")
                    elif iou_score >= 0.3:
                        print(f"    - Moderate agreement between ellipse and segmentation (IoU >= 0.3)")
                    else:
                        print(f"    - Poor agreement between ellipse and segmentation (IoU < 0.3)")

                    # Visualize overlap in threshold-specific folder
                    overlap_dir = os.path.join(output_base, threshold_folder, "overlap_comparison")
                    os.makedirs(overlap_dir, exist_ok=True)
                    overlap_output = os.path.join(overlap_dir, filename)
                    visualize_overlap(original_image, ellipse_mask, seg_mask, overlap_output)

                    print(f"\n  Color coding:")
                    print(f"    - Green: Ellipse only (from arrowhead detection)")
                    print(f"    - Red: Segmentation only (from process_image)")
                    print(f"    - Yellow: Overlap (agreement between both methods)")
                    print(f"{'='*60}")

                elif len(locs) == 2:
                    stats['line_drawn'] += 1
                    print(f"Drawing straight line from 2 points...")
                    draw_line_from_2_points(original_image, locs, output_filename)

                    # Also save process_image outputs for line case
                    contour_dir = os.path.join(output_base, threshold_folder, "contour_overlay")
                    mask_overlay_dir = os.path.join(output_base, threshold_folder, "mask_overlay")
                    drawing_dir = os.path.join(output_base, threshold_folder, "drawing")
                    skeleton_overlay_dir = os.path.join(output_base, threshold_folder, "skeleton")
                    colormap_overlay_dir = os.path.join(output_base, threshold_folder, "colormap_overlay")
                    ellipse_axis_overlay_dir = os.path.join(output_base, threshold_folder, "ellipse_axis_overlay")

                    for p in [contour_dir, mask_overlay_dir, drawing_dir, skeleton_overlay_dir,
                              colormap_overlay_dir, ellipse_axis_overlay_dir]:
                        os.makedirs(p, exist_ok=True)

                    # Process image to save to correct directories
                    process_image(image_path, contour_dir, mask_overlay_dir, drawing_dir,
                                 skeleton_overlay_dir, colormap_overlay_dir, ellipse_axis_overlay_dir,
                                 measurement_type)
                else:
                    print(f"Warning: Detected {len(locs)} points. Expected 2 or 4 points.")
                    print(f"No overlay will be drawn for this case.")

                # Clean up temp directory
                import shutil
                shutil.rmtree(os.path.join(output_base, "temp"), ignore_errors=True)

                print(f"\nSaved all outputs to: {threshold_folder}/")

            print(f"\n{'='*80}")
            print(f"Completed processing {filename}")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")
            stats['processing_errors'] += 1
            continue

    # Print summary statistics
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE - SUMMARY")
    print("="*80)
    print(f"Total images processed: {stats['total_images']}")
    print(f"  - Quality passed: {stats['quality_pass']}")
    print(f"  - Quality failed: {stats['quality_fail']}")
    print(f"  - Arrowheads detected: {stats['arrowheads_detected']}")
    print(f"  - No arrowheads: {stats['no_arrowheads']}")
    print(f"  - Ellipses drawn: {stats['ellipse_drawn']}")
    print(f"  - Lines drawn: {stats['line_drawn']}")
    print(f"  - Processing errors: {stats['processing_errors']}")
    print(f"\nSegmentation Threshold Results:")
    print(f"  - Above threshold: {stats['above_threshold']}")
    print(f"  - Below threshold: {stats['below_threshold']}")
    print(f"\nAll outputs saved to: {output_base}")
    print(f"  - Above threshold images: {output_base}/above_threshold/")
    print(f"  - Below threshold images: {output_base}/below_threshold/")
    print("="*80)

    return stats



# ---------------- MAIN ----------------
if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description="Fetal Ultrasound Image Processing Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Path to input image or folder containing images")
    parser.add_argument("--output", "-o", required=True, help="Output base directory for all results")
    parser.add_argument("--measurement-type", "-m", required=True,
                        choices=["01_abdominal_circumference", "02_bpd_hc_transthalamic", "03_femur_length"],
                        help="Type of measurement to perform")
    parser.add_argument("--brisque-threshold", type=float, default=60,
                        help="BRISQUE quality threshold (default: 60)")
    parser.add_argument("--niqe-threshold", type=float, default=7,
                        help="NIQE quality threshold (default: 7)")
    parser.add_argument("--seg-confidence-threshold", type=float, default=0.5,
                        help="Minimum segmentation confidence threshold (default: 0.5)")
    parser.add_argument("--seg-uncertainty-threshold", type=float, default=0.1,
                        help="Maximum segmentation uncertainty threshold (default: 0.1)")

    args = parser.parse_args()

    # Configuration from command line
    input_path = args.input
    output_base = args.output
    measurement_type = args.measurement_type
    BRISQUE_THRESHOLD = args.brisque_threshold
    NIQE_THRESHOLD = args.niqe_threshold
    SEG_CONFIDENCE_THRESHOLD = args.seg_confidence_threshold
    SEG_UNCERTAINTY_THRESHOLD = args.seg_uncertainty_threshold

    # YOLO configuration
    yolo_weight = "/home/htic/vasanth/fetalai_project/fetalai/yolo_arrowhead/yolov4-tiny-arrowhead.weights"
    yolo_cfg = "/home/htic/vasanth/fetalai_project/fetalai/yolo_arrowhead/yolov4-tiny-arrowhead.cfg"

    os.makedirs(output_base, exist_ok=True)

    # Check if input is a file or directory
    if os.path.isfile(input_path):
        # Single image processing
        input_image = input_path
        print(f"Processing single image: {input_image}")
        print(f"Measurement type: {measurement_type}")
        print(f"BRISQUE Threshold: {BRISQUE_THRESHOLD}")
        print(f"NIQE Threshold: {NIQE_THRESHOLD}")

        # Step 1: Assess Image Quality
        print("\n" + "="*60)
        print("STEP 1: ASSESSING IMAGE QUALITY")
        print("="*60)
        brisque_score, niqe_score = assess_image_quality(input_image)

        if brisque_score is None or niqe_score is None:
            print("Error: Could not assess image quality. Exiting.")
            exit(1)

        # Check quality thresholds
        quality_pass = True
        if brisque_score > BRISQUE_THRESHOLD:
            print(f"\nWarning: BRISQUE score ({brisque_score:.4f}) exceeds threshold ({BRISQUE_THRESHOLD})")
            quality_pass = False
        else:
            print(f"\nBRISQUE score ({brisque_score:.4f}) passes threshold ({BRISQUE_THRESHOLD})")

        if niqe_score > NIQE_THRESHOLD:
            print(f"Warning: NIQE score ({niqe_score:.4f}) exceeds threshold ({NIQE_THRESHOLD})")
            quality_pass = False
        else:
            print(f"NIQE score ({niqe_score:.4f}) passes threshold ({NIQE_THRESHOLD})")

        if not quality_pass:
            print("\nImage quality is below acceptable thresholds. Consider using a higher quality image.")
            print("Continuing with processing anyway...")
        else:
            print("\nImage quality is acceptable. Proceeding with detection.")

        # Step 2: Detect Arrowheads
        print("\n" + "="*60)
        print("STEP 2: DETECTING ARROWHEADS")
        print("="*60)

        img, locs, filename = detect_all(
            data_root=input_image,
            yolo_weight=yolo_weight,
            yolo_cfg=yolo_cfg,
            num_points=4,
            save_figs=True,
            dest=os.path.join(output_base, "yolo_detection"),
            single_image=True
        )

        if locs is None:
            print("\nNo arrowheads detected!")
            print("="*60)
            print("STEP 3: RUNNING PROCESS_IMAGE FOR NO-MARKING SCENARIO")
            print("="*60)

            # Create output directories
            contour_dir = os.path.join(output_base, "contour_overlay")
            mask_overlay_dir = os.path.join(output_base, "mask_overlay")
            drawing_dir = os.path.join(output_base, "drawing")
            skeleton_overlay_dir = os.path.join(output_base, "skeleton")
            colormap_overlay_dir = os.path.join(output_base, "colormap_overlay")
            ellipse_axis_overlay_dir = os.path.join(output_base, "ellipse_axis_overlay")

            for p in [contour_dir, mask_overlay_dir, drawing_dir, skeleton_overlay_dir,
                      colormap_overlay_dir, ellipse_axis_overlay_dir]:
                os.makedirs(p, exist_ok=True)

            # Process image without markers
            if os.path.exists(input_image):
                _, _, seg_confidence, seg_uncertainty = process_image(
                    input_image, contour_dir, mask_overlay_dir, drawing_dir,
                    skeleton_overlay_dir, colormap_overlay_dir, ellipse_axis_overlay_dir,
                    measurement_type
                )
                print(f"\nProcessing complete! Results saved to '{output_base}' directory")
                print(f"Segmentation Confidence: {seg_confidence:.4f}")
                print(f"Segmentation Uncertainty: {seg_uncertainty:.4f}")
            else:
                print(f"Error: Image file '{input_image}' not found!")
        else:
            print(f"\nArrowheads detected at locations: {locs}")
            print(f"Detection visualization saved to '{output_base}/yolo_detection'")

            # Step 3: Draw ellipse or line based on number of points
            print("\n" + "="*60)
            print("STEP 3: DRAWING OVERLAY BASED ON DETECTED POINTS")
            print("="*60)

            # Create output directory for marked overlays
            marked_overlay_dir = os.path.join(output_base, "marked_overlay")
            os.makedirs(marked_overlay_dir, exist_ok=True)

            # Load the original image
            original_image = cv2.imread(input_image)

            if original_image is None:
                print(f"Error: Could not load image from {input_image}")
            else:
                output_filename = os.path.join(marked_overlay_dir, os.path.basename(input_image))

                if len(locs) == 4:
                    print(f"Drawing ellipse with major and minor axes from 4 points...")
                    img_with_ellipse, ellipse_mask = draw_ellipse_from_4_points(original_image, locs, output_filename)

                    # Step 4: Run process_image to get segmentation mask
                    print("\n" + "="*60)
                    print("STEP 4: RUNNING PROCESS_IMAGE FOR SEGMENTATION COMPARISON")
                    print("="*60)

                    # Create output directories for process_image
                    contour_dir = os.path.join(output_base, "contour_overlay")
                    mask_overlay_dir = os.path.join(output_base, "mask_overlay")
                    drawing_dir = os.path.join(output_base, "drawing")
                    skeleton_overlay_dir = os.path.join(output_base, "skeleton")
                    colormap_overlay_dir = os.path.join(output_base, "colormap_overlay")
                    ellipse_axis_overlay_dir = os.path.join(output_base, "ellipse_axis_overlay")

                    for p in [contour_dir, mask_overlay_dir, drawing_dir, skeleton_overlay_dir,
                              colormap_overlay_dir, ellipse_axis_overlay_dir]:
                        os.makedirs(p, exist_ok=True)

                    # Process image to get segmentation mask
                    seg_mask, seg_image, seg_confidence, seg_uncertainty = process_image(
                        input_image, contour_dir, mask_overlay_dir, drawing_dir,
                        skeleton_overlay_dir, colormap_overlay_dir, ellipse_axis_overlay_dir,
                        measurement_type
                    )

                    # Display segmentation metrics
                    print(f"\nSegmentation Metrics:")
                    print(f"  Confidence: {seg_confidence:.4f}")
                    print(f"  Uncertainty: {seg_uncertainty:.4f}")

                    # Step 5: Compare masks and visualize overlap
                    print("\n" + "="*60)
                    print("STEP 5: COMPARING ELLIPSE AND SEGMENTATION MASKS")
                    print("="*60)

                    # Calculate overlap metrics
                    iou_score = calculate_iou(ellipse_mask, seg_mask)
                    dice_score = calculate_dice_coefficient(ellipse_mask, seg_mask)

                    print(f"\nOverlap Metrics:")
                    print(f"  IoU (Intersection over Union): {iou_score:.4f}")
                    print(f"  Dice Coefficient: {dice_score:.4f}")
                    print(f"\n  Interpretation:")
                    if iou_score >= 0.7:
                        print(f"    - Excellent agreement between ellipse and segmentation (IoU >= 0.7)")
                    elif iou_score >= 0.5:
                        print(f"    - Good agreement between ellipse and segmentation (IoU >= 0.5)")
                    elif iou_score >= 0.3:
                        print(f"    - Moderate agreement between ellipse and segmentation (IoU >= 0.3)")
                    else:
                        print(f"    - Poor agreement between ellipse and segmentation (IoU < 0.3)")

                    # Visualize overlap
                    overlap_dir = os.path.join(output_base, "overlap_comparison")
                    os.makedirs(overlap_dir, exist_ok=True)
                    overlap_output = os.path.join(overlap_dir, os.path.basename(input_image))
                    visualize_overlap(original_image, ellipse_mask, seg_mask, overlap_output)

                    print(f"\n  Color coding:")
                    print(f"    - Green: Ellipse only (from arrowhead detection)")
                    print(f"    - Red: Segmentation only (from process_image)")
                    print(f"    - Yellow: Overlap (agreement between both methods)")
                    print(f"{'='*60}")

                elif len(locs) == 2:
                    print(f"Drawing straight line from 2 points...")
                    draw_line_from_2_points(original_image, locs, output_filename)
                else:
                    print(f"Warning: Detected {len(locs)} points. Expected 2 or 4 points.")
                    print(f"No overlay will be drawn for this case.")

            print("\n" + "="*60)
            print("WORKFLOW COMPLETE")
            print("="*60)

    elif os.path.isdir(input_path):
        # Folder processing
        print(f"Processing folder: {input_path}")
        stats = process_folder(input_path, output_base, measurement_type, BRISQUE_THRESHOLD, NIQE_THRESHOLD,
                               SEG_CONFIDENCE_THRESHOLD, SEG_UNCERTAINTY_THRESHOLD)
    else:
        print(f"Error: Input path '{input_path}' is neither a file nor a directory!")
        exit(1)


#######################

#   python inf_2.py \
#     --input /path/to/images/ \
#     --output /path/to/output/ \
#     --measurement-type 02_bpd_hc_transthalamic \
#     --seg-confidence-threshold 0.7 \
#     --seg-uncertainty-threshold 0.05
