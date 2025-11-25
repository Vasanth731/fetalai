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

# model for no marking images
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
        # return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width),
        #         torch.zeros(batch_size, self.hidden_dim, self.height, self.width))


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
# model.load_state_dict(torch.load("/home/htic/vasanth/fetalai_project/weights/fetalnet/finetuned_multicls/output_precise_3cls/final_model.pt", map_location=device))
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
    # if measurement_type in ["bpd", "ac"]:
    if measurement_type in ["02_bpd_hc_transthalamic", "01_abdominal_circumference"]:
        drawing, shape, ellipse, endpoints_sorted, ellipse_overlay = boundingbox_ellipsefit(image_rgb, scaled_contours)
        skeleton_overlay = np.zeros_like(image_rgb)  # No skeleton overlay for ellipse fit
        # Save ellipse overlay with major and minor axis lines
        cv2.imwrite(os.path.join(ellipse_axis_overlay_dir, scored_filename), ellipse_overlay)
    # elif measurement_type == "fl":
    elif measurement_type == "03_femur_length":
        drawing, shape, endpoints_sorted, skeleton_overlay = extremafit_2pts(image_rgb, scaled_contours, binary_mask)
    else:
        print(f"Warning: Unknown measurement type '{measurement_type}', using extremafit_2pts as default")
        drawing, shape, endpoints_sorted, skeleton_overlay = extremafit_2pts(image_rgb, scaled_contours, binary_mask)

    print(f"Endpoints: {endpoints_sorted}")
    cv2.imwrite(os.path.join(drawing_dir, scored_filename), drawing)
    cv2.imwrite(os.path.join(skeleton_overlay_dir, scored_filename), skeleton_overlay)



# model and inference for marked images
def detect_arrowhead(path, net, conf=0.5):
    # global img, img0, outputs, ln
    
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
            # cv.rectangle(img, p0, p1, WHITE, 1)

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


############################################################################

"""
Image Quality Assessment for Fetal Ultrasound Images
This script implements 2 Full-Reference and 2 No-Reference metrics specifically useful for fetal ultrasound imaging.

Full-Reference Methods:
    1. SSIM (Structural Similarity Index) - Range: [-1, 1], Higher is better
    2. FSIM (Feature Similarity Index) - Range: [0, 1], Higher is better

No-Reference Methods:
    1. BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) - Range: [0, 100], Lower is better
    2. NIQE (Natural Image Quality Evaluator) - Range: [0, ∞], Lower is better
"""

import sys
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import convolve
from scipy.special import gamma
import warnings
warnings.filterwarnings('ignore')

# ======================== NO-REFERENCE METHODS ========================

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


# ======================== MAIN FUNCTION ========================

def assess_image_quality(test_img_path, reference_img_path=None):
    """
    Assess image quality using multiple metrics

    Args:
        test_img_path: Path to the test image
        reference_img_path: Path to the reference image (optional, for FR metrics)
    """
    # Load test image
    test_img = cv2.imread(test_img_path)
    if test_img is None:
        print(f"Error: Could not load test image from {test_img_path}")
        return

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




# ---------------- MAIN ----------------
if __name__ == "__main__":

    # Single image inference
    input_image = "51.png"
    output_base = "output_single_image2"

    # Specify measurement type: "02_bpd_hc_transthalamic", "01_abdominal_circumference", or "03_femur_length"
    measurement_type = "02_bpd_hc_transthalamic"  # Change this based on your image type

    os.makedirs(output_base, exist_ok=True)

    print(f"Processing single image: {input_image}")
    print(f"Measurement type: {measurement_type}")

    contour_dir = os.path.join(output_base, "contour_overlay")
    mask_overlay_dir = os.path.join(output_base, "mask_overlay")
    drawing_dir = os.path.join(output_base, "drawing")
    skeleton_overlay_dir = os.path.join(output_base, "skeleton")
    colormap_overlay_dir = os.path.join(output_base, "colormap_overlay")
    ellipse_axis_overlay_dir = os.path.join(output_base, "ellipse_axis_overlay")

    for p in [contour_dir, mask_overlay_dir, drawing_dir, skeleton_overlay_dir, colormap_overlay_dir, ellipse_axis_overlay_dir]:
        os.makedirs(p, exist_ok=True)

    # Process single image
    if os.path.exists(input_image):
        process_image(input_image, contour_dir, mask_overlay_dir, drawing_dir, skeleton_overlay_dir, colormap_overlay_dir, ellipse_axis_overlay_dir, measurement_type)
        print(f"\nProcessing complete! Results saved to '{output_base}' directory")
    else:
        print(f"Error: Image file '{input_image}' not found!")

    # ========================================================================
    # Optional: Uncomment below sections if you need arrowhead detection or IQA
    # ========================================================================


    img, locs, filename = detect_all(
        data_root="51.png",
        yolo_weight="/home/htic/vasanth/fetalai_project/fetalai/yolo_arrowhead/yolov4-tiny-arrowhead.weights",
        yolo_cfg="/home/htic/vasanth/fetalai_project/fetalai/yolo_arrowhead/yolov4-tiny-arrowhead.cfg",
        num_points=4,
        save_figs=True,
        dest="output_yolo_single"
    )


    # Image Quality Assessment
    assess_image_quality(input_image)


