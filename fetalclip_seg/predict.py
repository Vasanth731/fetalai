'''
Code for running inference per image basis.
'''
import cv2
import torch
import torch.nn as nn
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from PIL import Image,ImageOps
import albumentations as A
import open_clip
from utils_get_embeddings import (
    EncoderWrapper,
    get_test_embeddings,
    get_train_val_embeddings,
)
# from utils import LitModel

with open('config.json', 'r') as file:
    config = json.load(file)

OVERLAY_DIR='./infer_fl/overlays'
MASK_DIR='./infer_fl/masks'
# CKPT_PATH = "/home/htic/Documents/FetalCLIP/segmentation/final_data/segmentation/AC/fetalclip/0_0/epoch=19-step=2120.ckpt"
# CKPT_PATH = "/home/htic/Documents/FetalCLIP/segmentation/final_data/segmentation/BPD/fetalclip/0_0/epoch=19-step=3420.ckpt"
CKPT_PATH = "/home/htic/Documents/FetalCLIP/segmentation/final_data/segmentation/FL/fetalclip/0_0/epoch=13-step=1610.ckpt"
DIR_DATA = "/home/htic/anu/sononet_cusp/in/mediscan_video/FL"
SAVE_IMAGE_SIZE = 224


LIST_CLASSES = ['Femur'] # head
NUM_CLASSES  = len(LIST_CLASSES)
INIT_FILTERS = 32

PATH_FETALCLIP_WEIGHT = config['paths']['path_fetalclip_weight']
PATH_FETALCLIP_CONFIG = config['paths']['path_fetalclip_config']
ARCH_NAME = 'FetalCLIP'

preprocessing = A.Compose([
    A.Resize(
        SAVE_IMAGE_SIZE, SAVE_IMAGE_SIZE, interpolation=cv2.INTER_CUBIC,
        mask_interpolation=0
    ),
])

'''
REFERENCES:
- https://github.com/tamasino52/UNETR/blob/main/unetr.py#L171
'''

class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1):
        super().__init__()
        self.block = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0, groups=groups)

    def forward(self, x):
        return self.block(x)

class SingleConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1):
        super().__init__()
        self.block = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2), groups=groups)

    def forward(self, x):
        return self.block(x)

class Conv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2DBlock(in_planes, in_planes, kernel_size, groups=in_planes),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(True),
            SingleConv2DBlock(in_planes, out_planes, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)

class Deconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_planes, in_planes, groups=in_planes),
            SingleConv2DBlock(in_planes, in_planes, kernel_size, groups=in_planes),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(True),
            SingleConv2DBlock(in_planes, out_planes, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)

class SingleDWConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_planes, in_planes, groups=in_planes),
            SingleConv2DBlock(in_planes, out_planes, 1),
        )

    def forward(self, x):
        return self.block(x)

class UNETR(nn.Module):
    def __init__(self, transformer_width, output_dim, input_dim, init_filters):
        super().__init__()

        self.decoder0 = \
            nn.Sequential(
                Conv2DBlock(input_dim, init_filters, 3),
                Conv2DBlock(init_filters, init_filters, 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv2DBlock(transformer_width, 8*init_filters),
                Deconv2DBlock(8*init_filters, 4*init_filters),
                Deconv2DBlock(4*init_filters, 2*init_filters)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv2DBlock(transformer_width, 8*init_filters),
                Deconv2DBlock(8*init_filters, 4*init_filters),
            )

        self.decoder9 = \
            Deconv2DBlock(transformer_width, 8*init_filters)

        self.decoder12_upsampler = \
            SingleDWConv2DBlock(transformer_width, 8*init_filters)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv2DBlock(16*init_filters, 8*init_filters),
                Conv2DBlock(8*init_filters, 8*init_filters),
                Conv2DBlock(8*init_filters, 8*init_filters),
                SingleDWConv2DBlock(8*init_filters, 4*init_filters)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv2DBlock(8*init_filters, 4*init_filters),
                Conv2DBlock(4*init_filters, 4*init_filters),
                SingleDWConv2DBlock(4*init_filters, 2*init_filters)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv2DBlock(4*init_filters, 2*init_filters),
                Conv2DBlock(2*init_filters, 2*init_filters),
                SingleDWConv2DBlock(2*init_filters, init_filters)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv2DBlock(2*init_filters, init_filters),
                Conv2DBlock(init_filters, init_filters),
                SingleConv2DBlock(init_filters, output_dim, 1)
            )
    
    def forward(self, x):
        z0, z3, z6, z9, z12 = x
        
        # print(z0.shape, z3.shape, z6.shape, z9.shape, z12.shape)
        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        # print(z0.shape, z3.shape, z6.shape, z9.shape, z12.shape)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))

        return output

class LitModel(nn.Module):
    def __init__(self, transformer_width, num_classes, input_dim, init_filters):
        super().__init__()
        self.model = UNETR(transformer_width, num_classes, input_dim, init_filters)
    
    def forward(self, x):
        x = self.model(x)
        return x

def make_image_square_with_zero_padding(image):
    orig_img = image
    width, height = image.size

    # Determine the size of the square
    max_side = max(width, height)

    # Create a new square image with black padding (0 for black in RGB or L modes)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    if image.mode == "RGB":
        padding_color = (0, 0, 0)  # Black for RGB images
    elif image.mode == "L":
        padding_color = 0  # Black for grayscale images

    # Create a new square image
    new_image = Image.new(image.mode, (max_side, max_side), padding_color)

    # Calculate padding
    padding_left = (max_side - width) // 2
    padding_top = (max_side - height) // 2

    # Paste the original image in the center of the new square image
    new_image.paste(image, (padding_left, padding_top))

    return new_image, orig_img

def setup_model(ckpt_path):
    # Load model configuration
    with open(PATH_FETALCLIP_CONFIG, "r") as file:
        config_fetalclip = json.load(file)
    open_clip.factory._MODEL_CONFIGS[ARCH_NAME] = config_fetalclip

    model, _, preprocess_img = open_clip.create_model_and_transforms(ARCH_NAME, pretrained=PATH_FETALCLIP_WEIGHT)
    model.visual.eval()
    
    model_name = ARCH_NAME
    encoder = EncoderWrapper(model.visual)
    encoder.eval()
    encoder = encoder.cuda()

    # load weights
    # model = LitModel.load_from_checkpoint(
    #     ckpt_path,
    #     transformer_width=encoder.transformer.width,
    #     num_classes=NUM_CLASSES, input_dim=3,init_filters=INIT_FILTERS
    # )
    model = LitModel(
        transformer_width=encoder.transformer.width,
        num_classes=NUM_CLASSES, input_dim=3,init_filters=INIT_FILTERS
    )
    ckpt = torch.load(CKPT_PATH)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().cuda()

    return encoder,model, preprocess_img

def run_infer_img(img, encoder, model, preprocess_img):
    img_np = np.array(img)
    img_resized = preprocessing(image=img_np)['image'] #resized into (224,224)
    img_resized = Image.fromarray(img_resized)
    img_t = preprocess_img(img_resized)
    img = img_t.unsqueeze(0).cuda()
    with torch.no_grad():
        z = encoder(img) # CLIP-Visual encoder

    embs = [z[k] for k in ['z3', 'z6', 'z9', 'z12']]
    pred = model([img, *embs]) # UNETR
    sigmoid_mask = torch.sigmoid(pred)
    pred = (sigmoid_mask > 0.5).float()
    pred_mask = pred.squeeze().detach().cpu().numpy()
    return pred_mask, img_resized


def extract_original_from_padded(padded_image, orig_size):
    padded_width, padded_height = padded_image.size
    orig_height,orig_width = orig_size
    # Calculate the crop box
    left = (padded_width - orig_width) // 2
    top = (padded_height - orig_height) // 2
    right = left + orig_width
    bottom = top + orig_height

    # Crop the original image region
    original_image = padded_image.crop((left, top, right, bottom))
    return original_image

# def save_prediction(img_path,img,pred,orig_size, pad_size):
#     # Convert pred_mask to PIL and resize to match original image
#     mask = Image.fromarray((pred * 255).astype(np.uint8))  # scale binary mask

#     # mask = mask.resize(pad_size, resample=Image.NEAREST)
#     # img = img.resize(pad_size, resample=Image.NEAREST)
#     # mask = extract_original_from_padded(mask, orig_size)
#     # img = extract_original_from_padded(img, orig_size)

#     # Save mask
#     mask_save_path = os.path.join(MASK_DIR, os.path.basename(img_path))
#     mask.save(mask_save_path)

#     # Convert grayscale orig_img to RGB
#     orig_img_rgb = img.convert("RGBA")

#     # Convert mask to binary mask (non-zero becomes red)
#     mask_arr = np.array(mask)
#     mask_binary = (mask_arr > 0).astype(np.uint8)

#     # Create red overlay with 0.5 alpha where mask is active
#     overlay = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 4), dtype=np.uint8)
#     overlay[..., 0] = 255  # Red channel
#     overlay[..., 3] = mask_binary * int(255 * 0.5)  # Alpha channel = 127 if mask is 1

#     # Convert overlay to Pillow Image
#     overlay_img = Image.fromarray(overlay, mode="RGBA")

#     # Blend
#     blended = Image.alpha_composite(orig_img_rgb, overlay_img)
#     overlay_save_path = os.path.join(OVERLAY_DIR, os.path.basename(img_path))
#     blended.save(overlay_save_path)

# def run_infer_folder(folder_path=DIR_DATA):
#     list_images = [p for p in os.listdir(folder_path) if 'Annotation' not in p]
#     encoder, model, preprocess_img = setup_model(CKPT_PATH)

#     os.makedirs(OVERLAY_DIR, exist_ok=True)
#     os.makedirs(MASK_DIR, exist_ok=True)
#     # for filename in tqdm(list_images):
#     for filename in tqdm(list_images):
#         pid = str(int(filename.split('_')[0]))
#         imgname = filename.split('.')[0]
#         img_path= imgname+'.png'

#         img = Image.open(os.path.join(folder_path, f'{imgname}.jpg'))
#         img_padded, orig_img = make_image_square_with_zero_padding(img)

#         pred_mask, img_resized = run_infer_img(img_padded, encoder, model, preprocess_img)
#         save_prediction(img_path,img_resized,pred_mask,img.size, img_padded.size)

def save_prediction(img_path, img, pred, orig_size, pad_size):
    mask = Image.fromarray((pred * 255).astype(np.uint8))
    
    # Save binary mask
    mask_filename = os.path.splitext(os.path.basename(img_path))[0] + ".png"
    mask_save_path = os.path.join(MASK_DIR, mask_filename)
    mask.save(mask_save_path)

    # Convert original image to RGBA for overlay
    orig_img_rgb = img.convert("RGBA")

    # Create red overlay where mask is active
    mask_arr = np.array(mask)
    mask_binary = (mask_arr > 0).astype(np.uint8)

    overlay = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 4), dtype=np.uint8)
    overlay[..., 0] = 255  # Red
    overlay[..., 3] = mask_binary * int(255 * 0.5)  # Alpha

    overlay_img = Image.fromarray(overlay, mode="RGBA")
    blended = Image.alpha_composite(orig_img_rgb, overlay_img)

    # Save overlay as PNG
    overlay_filename = os.path.splitext(os.path.basename(img_path))[0] + ".png"
    overlay_save_path = os.path.join(OVERLAY_DIR, overlay_filename)
    blended.save(overlay_save_path)


def run_infer_folder(folder_path=DIR_DATA):
    list_images = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]

    encoder, model, preprocess_img = setup_model(CKPT_PATH)

    os.makedirs(OVERLAY_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

    for filename in tqdm(list_images):
        img_path = os.path.join(folder_path, filename)

        # Open image
        img = Image.open(img_path).convert("RGB")  # ensure it's RGB

        # Pad image to square
        img_padded, orig_img = make_image_square_with_zero_padding(img)

        # Run inference
        pred_mask, img_resized = run_infer_img(img_padded, encoder, model, preprocess_img)

        # Save prediction and overlay
        save_prediction(filename, img_resized, pred_mask, img.size, img_padded.size)


def run_infer_video(video_path):
    encoder, model, preprocess_img = setup_model(CKPT_PATH)
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Loop through each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame[:,250:-110,:]

        # Convert BGR (OpenCV) to RGB (Pillow)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to Pillow Image
        img = Image.fromarray(frame_rgb)
        img_padded, orig_img = make_image_square_with_zero_padding(img)
        pred_mask, img_resized = run_infer_img(img_padded, encoder, model, preprocess_img)

        # Show the frame in an OpenCV window
        cv2.imshow('Video Frame', frame)

        # Press 'q' to quit early
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_infer_folder(DIR_DATA)
    # run_infer_video("../../../../data/in-house/2002_2nd TRI_TARGET/1/2002_21_20250326_161033.mp4")
