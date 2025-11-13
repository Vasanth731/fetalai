import os
import json
import shutil

import cv2
import numpy as np
import albumentations as A

from tqdm import tqdm
from PIL import Image

with open('config.json', 'r') as file:
    config = json.load(file)

DIR_DATA = config['paths']['dir_data']
PATH_TRAIN_VAL_SPLIT = config['paths']['path_train_val_split']
PATH_TEST_SPLIT = config['paths']['path_test_split']
SAVE_PREPROCESSED_DIR = config['paths']['dir_preprocessed']

SAVE_IMAGE_SIZE = 224
N_AUGMENTATIONS = 5

with open(PATH_TRAIN_VAL_SPLIT, 'r') as file:
    dict_list_pid = json.load(file)

with open(PATH_TEST_SPLIT, 'r') as file:
    list_test_pid = json.load(file)

list_test_pid = list(list_test_pid.keys())

AUGMENTATION = A.Compose([
    A.ColorJitter(0.2, 0.2, 0.2, 0.2, p=0.5),
    A.CLAHE(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.2,
        scale_limit=0.0,
        rotate_limit=20,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT, value=0, p=1.
    ),
])

preprocessing = A.Compose([
    A.Resize(
        SAVE_IMAGE_SIZE, SAVE_IMAGE_SIZE, interpolation=cv2.INTER_CUBIC,
        mask_interpolation=0, always_apply=True
    ),
])

if os.path.exists(SAVE_PREPROCESSED_DIR):
    shutil.rmtree(SAVE_PREPROCESSED_DIR)
os.makedirs(os.path.join(SAVE_PREPROCESSED_DIR, 'train'))
os.makedirs(os.path.join(SAVE_PREPROCESSED_DIR, 'val'))
os.makedirs(os.path.join(SAVE_PREPROCESSED_DIR, 'test'))

def make_image_square_with_zero_padding(image):
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

    return new_image

def fill_contour(img):
        assert len(img.shape) == 2
        assert len(np.unique(img)) == 2
        assert img.dtype == np.uint8

        out = img.copy()
        contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)

        return out
list_pid_train = []
list_pid_val = []
for _, v in dict_list_pid.items():
    list_pid_train.extend(v[0])
    list_pid_val.extend(v[1])
list_pid_train = list(set(list_pid_train))
list_pid_val   = list(set(list_pid_val))

len(list_pid_train), len(list_pid_val)
list_pid_train = [str(int(l)) for l in list_pid_train]
list_pid_val   = [str(int(l)) for l in list_pid_val]
list_test_pid  = [str(int(l)) for l in list_test_pid]


list_images = [p for p in os.listdir(DIR_DATA) if 'Annotation' not in p]
for filename in tqdm(list_images):
    pid = str(int(filename.split('_')[0]))
    imgname = filename.split('.')[0]
    if pid not in list_pid_train:
        continue
    
    img = Image.open(os.path.join(DIR_DATA, f'{imgname}.png'))
    ann = Image.open(os.path.join(DIR_DATA, f'{imgname}_Annotation.png')).convert("L")
    
    assert np.array(ann).shape[:2] == np.array(img).shape[:2]

    img = make_image_square_with_zero_padding(img)
    ann = make_image_square_with_zero_padding(ann)

    img = np.array(img)
    ann = np.array(ann)
    ann = (ann > 0).astype(np.uint8) * 255
    # ann = fill_contour(ann)

    assert ann.shape[:2] == img.shape[:2]

    data = {'img': img.copy(), 'label': ann.copy()}
    
    for i in range(N_AUGMENTATIONS):
        img = data['img'].copy()
        ann = data['label'].copy()

        if AUGMENTATION:
            augmented = AUGMENTATION(image=img, mask=ann)
            img = augmented['image']
            ann = augmented['mask']
        
        img_ann = preprocessing(image=img, mask=ann)
        img = img_ann['image']
        ann = img_ann['mask']

        np.savez(
            os.path.join(SAVE_PREPROCESSED_DIR, "train", f"{filename.split('.')[0]}_{i}.npz"),
            img=img, ann=ann,
        )

        img = Image.fromarray(img)
        ann = Image.fromarray(ann)

for filename in tqdm(list_images):
    pid = str(int(filename.split('_')[0]))
    imgname = filename.split('.')[0]
    if pid not in list_pid_val:
        continue
    img = Image.open(os.path.join(DIR_DATA, f'{imgname}.png'))
    ann = Image.open(os.path.join(DIR_DATA, f'{imgname}_Annotation.png')).convert("L")
    
    assert np.array(ann).shape[:2] == np.array(img).shape[:2]

    img = make_image_square_with_zero_padding(img)
    ann = make_image_square_with_zero_padding(ann)

    img = np.array(img)
    ann = np.array(ann)
    ann = (ann > 0).astype(np.uint8) * 255
    # ann = fill_contour(ann)

    assert ann.shape[:2] == img.shape[:2]
    
    img_ann = preprocessing(image=img, mask=ann)
    img = img_ann['image']
    ann = img_ann['mask']

    np.savez(
        os.path.join(SAVE_PREPROCESSED_DIR, "val", f"{filename.split('.')[0]}.npz"),
        img=img, ann=ann,
    )

    img = Image.fromarray(img)
    ann = Image.fromarray(ann)

for filename in tqdm(list_images):
    pid = str(int(filename.split('_')[0]))
    imgname = filename.split('.')[0]
    if pid not in list_test_pid:
        continue
    img = Image.open(os.path.join(DIR_DATA, f'{imgname}.png'))
    ann = Image.open(os.path.join(DIR_DATA, f'{imgname}_Annotation.png')).convert("L")
    
    assert np.array(ann).shape[:2] == np.array(img).shape[:2]

    img = make_image_square_with_zero_padding(img)
    ann = make_image_square_with_zero_padding(ann)

    img = np.array(img)
    ann = np.array(ann)
    ann = (ann > 0).astype(np.uint8) * 255

    # ann = fill_contour(ann)

    assert ann.shape[:2] == img.shape[:2]
    
    img_ann = preprocessing(image=img, mask=ann)
    img = img_ann['image']
    ann = img_ann['mask']

    np.savez(
        os.path.join(SAVE_PREPROCESSED_DIR, "test", f"{filename.split('.')[0]}.npz"),
        img=img, ann=ann,
    )

    img = Image.fromarray(img)
    ann = Image.fromarray(ann)
print(len(os.listdir('preprocessed_data/train')))
print(len(os.listdir('preprocessed_data/val')))
print(len(os.listdir('preprocessed_data/test')))
 
