import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch

import os
import json

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as T
import albumentations as A
import open_clip
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils

from PIL import Image

with open('config.json', 'r') as file:
    config = json.load(file)

DIR_TRAIN = os.path.join(config['paths']['dir_preprocessed'], 'train')
DIR_VAL = os.path.join(config['paths']['dir_preprocessed'], 'val')
DIR_TEST = os.path.join(config['paths']['dir_preprocessed'], 'test')

PATH_TRAIN_VAL_SPLIT = config['paths']['path_train_val_split']
PATH_TEST_SPLIT = config['paths']['path_test_split']
DIR_SAVE_CSV_EXP = config['paths']['dir_experiment_logs']

NUM_WORKERS = config['params']['num_workers']
BATCH_SIZE = config['params']['batch_size']
MAX_EPOCHS = config['params']['max_epochs']

IMG_SIZE = 224
INIT_FILTERS = 32
N_RUNS_PER_EXP = 2
CHECK_VAL_EVERY_N_EPOCH = 1
PIN_MEMORY = True

with open(PATH_TRAIN_VAL_SPLIT, 'r') as file:
    DICT_LIST_PID = json.load(file)

with open('test_id.json', 'r') as file:
    LIST_TEST_PID = json.load(file)
LIST_TEST_PID = list(LIST_TEST_PID.keys())

# MULTICLASS SEGMENTATION CONFIGURATION
# Background (0) + BPD (1) + AC (2) + FL (3)
# Total 4 classes for multiclass segmentation
LIST_CLASSES = ['background', 'BPD', 'AC', 'FL']
NUM_CLASSES  = len(LIST_CLASSES)  # 4 classes total

os.makedirs(DIR_SAVE_CSV_EXP, exist_ok=True)

class DatasetHC18(torch.utils.data.Dataset):
    def __init__(
            self,
            root,
            list_patients,
            preprocess_img,
            preprocess_mask,
            dict_embeddings=None,
        ):

        self.preprocess_img = preprocess_img
        self.preprocess_mask = preprocess_mask
        self.dict_embeddings = dict_embeddings

        # Handle patient IDs with parentheses (e.g., "30415169(1)") and non-numeric IDs
        processed_patients = []
        for pid in list_patients:
            try:
                # Try to extract numeric part before parenthesis if present
                if '(' in pid:
                    pid = pid.split('(')[0]
                # Convert to int and back to string to normalize
                processed_patients.append(str(int(pid)))
            except ValueError:
                # Keep original ID if it's not convertible to int (e.g., "ANID")
                processed_patients.append(str(pid))
        list_patients = processed_patients

        self.data = []
        for filename in os.listdir(root):
            pid_raw = filename.split('_')[0]
            try:
                # Extract numeric part before parenthesis if present
                if '(' in pid_raw:
                    pid_raw = pid_raw.split('(')[0]
                pid = str(int(pid_raw))
            except ValueError:
                # Keep original if not convertible
                pid = str(pid_raw)

            if pid not in list_patients:
                continue

            if self.dict_embeddings:
                self.data.append((
                    os.path.join(root, filename),
                    dict_embeddings[filename.split('.')[0]],
                ))
            else:
                self.data.append(os.path.join(root, filename))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.dict_embeddings:
            data = np.load(self.data[index][0])
            embs = self.data[index][1]
        else:
            data = np.load(self.data[index])

        img = data['img']
        ann = data['ann']

        img = Image.fromarray(img)
        ann = Image.fromarray(ann)

        img = self.preprocess_img(img)
        ann = self.preprocess_mask(ann)

        # For multiclass segmentation: ann should be (H, W) with class indices
        # Remove the channel dimension added by ToTensor if present
        if ann.dim() == 3 and ann.shape[0] == 1:
            ann = ann.squeeze(0)  # (1, H, W) -> (H, W)

        # FIX: ToTensor divides by 255, so we need to multiply back to get original class indices
        # This converts 0.0, 0.004, 0.008, 0.012 back to 0, 1, 2, 3
        ann = (ann * 255).round()

        # Convert to long tensor for cross-entropy loss
        ann = ann.long()

        if self.dict_embeddings:
            return img, ann, embs
        else:
            return img, ann


class LitModel(pl.LightningModule):
    def __init__(self, transformer_width, num_classes, input_dim, init_filters):
        super().__init__()
        self.num_classes = num_classes
        self.model = UNETR(transformer_width, num_classes, input_dim, init_filters)
        # MULTICLASS: Use 'multiclass' mode for multiclass segmentation
        # For binary segmentation (NUM_CLASSES=1), use mode='binary'
        # For multiclass segmentation (NUM_CLASSES>1), use mode='multiclass'
        if num_classes == 1:
            self.criterion = smp.losses.DiceLoss(mode='binary', from_logits=True)
        else:
            self.criterion = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        self.validation_step_outputs = []
        self.validation_step_inputs = []
    
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y, embs = batch

        pred = self.forward([x, *embs])

        loss = self.criterion(pred, y)

        # MULTICLASS: Compute metrics using per-class scores
        if self.num_classes == 1:
            activation = 'sigmoid'
            dsc = smp_utils.metrics.Fscore(activation=activation)(pred, y)
        else:
            # For multiclass, compute per-class Dice and average (excluding background)
            # Convert predictions to probabilities
            pred_probs = torch.softmax(pred, dim=1)  # (B, C, H, W)
            # Convert targets to one-hot encoding
            targets_onehot = F.one_hot(y.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()  # (B, C, H, W)

            # Compute Dice per class
            class_dsc = []
            for c in range(1, self.num_classes):  # Skip background (class 0)
                tp = torch.sum(pred_probs[:, c] * targets_onehot[:, c])
                fp = torch.sum(pred_probs[:, c] * (1 - targets_onehot[:, c]))
                fn = torch.sum((1 - pred_probs[:, c]) * targets_onehot[:, c])
                dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
                class_dsc.append(dice)
            dsc = torch.mean(torch.stack(class_dsc))

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_dsc', dsc, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_nb):
        x, y, embs = batch
        
        pred = self.forward([x, *embs])
        
        self.validation_step_outputs.append((pred,y))
        self.validation_step_inputs.append(x)
        return pred, y
    
    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def save_overlay_images(self,orig_images, pred_masks, target_masks, addendum, save_dir='interim_output'):
        os.makedirs(save_dir, exist_ok=True)
        B = orig_images.size(0)
        alpha=0.5

        # MULTICLASS: Define colormap for different classes
        # Each class gets a unique color (RGB values)
        class_colors = [
            [0, 0, 0],      # Class 0: Black (background)
            [1, 0, 0],      # Class 1: Red
            [0, 1, 0],      # Class 2: Green
            [0, 0, 1],      # Class 3: Blue
            [1, 1, 0],      # Class 4: Yellow
            [1, 0, 1],      # Class 5: Magenta
            [0, 1, 1],      # Class 6: Cyan
            [1, 0.5, 0],    # Class 7: Orange
            [0.5, 0, 1],    # Class 8: Purple
            [0, 0.5, 0.5],  # Class 9: Teal
        ]

        for i in range(B):
            img = orig_images[i].detach().cpu()
            pred = pred_masks[i].detach().cpu()
            target = target_masks[i].detach().cpu()

            # Remove any extra dimensions
            if pred.dim() == 3:
                pred = pred.squeeze(0)
            if target.dim() == 3:
                target = target.squeeze(0)

            # Debug: Print mask statistics (only for first batch)
            if i == 0 and addendum.startswith('val_0'):
                print(f"[DEBUG] Target unique values: {torch.unique(target)}")
                print(f"[DEBUG] Pred unique values: {torch.unique(pred)}")
                print(f"[DEBUG] Target shape: {target.shape}, Pred shape: {pred.shape}")

            # Denormalize image from CLIP normalization (ImageNet stats)
            # CLIP uses mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            img = img * std + mean  # Denormalize
            img = torch.clamp(img, 0, 1)  # Clamp to [0, 1]

            # Convert to numpy [H, W, 3]
            img_np = img.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)

            # Create colored mask overlays for multiclass
            target_np = target.numpy().astype(np.int32)
            pred_np = pred.numpy().astype(np.int32)

            # Initialize colored masks
            target_colored = np.zeros((*target_np.shape, 3))
            pred_colored = np.zeros((*pred_np.shape, 3))

            # Apply colors for each class
            for class_idx in range(min(len(class_colors), self.num_classes + 1)):
                target_colored[target_np == class_idx] = class_colors[class_idx]
                pred_colored[pred_np == class_idx] = class_colors[class_idx]

            # Blend with alpha (skip background - class 0)
            overlay_gt = img_np.copy()
            overlay_pred = img_np.copy()

            for class_idx in range(1, min(len(class_colors), self.num_classes + 1)):
                mask_gt = (target_np == class_idx)
                mask_pred = (pred_np == class_idx)
                if mask_gt.any():
                    overlay_gt[mask_gt] = img_np[mask_gt] * (1 - alpha) + np.array(class_colors[class_idx]) * alpha
                if mask_pred.any():
                    overlay_pred[mask_pred] = img_np[mask_pred] * (1 - alpha) + np.array(class_colors[class_idx]) * alpha

            # Concatenate
            combined = np.concatenate((overlay_gt, overlay_pred), axis=1)
            combined = np.clip(combined, 0, 1)

            # Save
            plt.imsave(os.path.join(save_dir, f"{addendum}_{i}.png"), combined)

    def on_validation_epoch_end(self):
        preds = []
        targets = []

        for i,outs in enumerate(self.validation_step_outputs):
            preds.append(outs[0])
            targets.append(outs[1])
            input = self.validation_step_inputs[i]

            # MULTICLASS: Apply appropriate activation and get predictions
            if self.num_classes == 1:
                pred = torch.sigmoid(outs[0])
                pred = (pred > 0.5).float()
                target = outs[1].float().unsqueeze(1)  # Add channel dimension for binary
            else:
                pred = torch.softmax(outs[0], dim=1)  # (B, C, H, W)
                pred = torch.argmax(pred, dim=1)  # (B, H, W)
                target = outs[1]  # Already (B, H, W)

            self.save_overlay_images(input, pred, target, "val_"+str(i))

        preds = torch.cat(preds)
        targets = torch.cat(targets)

        loss = self.criterion(preds, targets)

        # MULTICLASS: Compute metrics using per-class scores
        if self.num_classes == 1:
            activation = 'sigmoid'
            dsc = smp_utils.metrics.Fscore(activation=activation)(preds.cpu(), targets.cpu())
        else:
            # For multiclass, compute per-class Dice and average (excluding background)
            # Convert predictions to probabilities
            pred_probs = torch.softmax(preds, dim=1)  # (N, C, H, W)
            # Convert targets to one-hot encoding
            targets_onehot = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()  # (N, C, H, W)

            # Compute Dice per class
            class_dsc = []
            for c in range(1, self.num_classes):  # Skip background (class 0)
                tp = torch.sum(pred_probs[:, c] * targets_onehot[:, c])
                fp = torch.sum(pred_probs[:, c] * (1 - targets_onehot[:, c]))
                fn = torch.sum((1 - pred_probs[:, c]) * targets_onehot[:, c])
                dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
                class_dsc.append(dice)
            dsc = torch.mean(torch.stack(class_dsc))
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dsc', dsc, on_step=False, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.clear()
        self.validation_step_inputs.clear()

    def on_test_epoch_end(self):
        preds = []
        targets = []

        for i,outs in enumerate(self.validation_step_outputs):
            preds.append(outs[0])
            targets.append(outs[1])
            input = self.validation_step_inputs[i]

            # MULTICLASS: Apply appropriate activation and get predictions
            if self.num_classes == 1:
                pred = torch.sigmoid(outs[0])
                pred = (pred > 0.5).float()
                target = outs[1].float().unsqueeze(1)  # Add channel dimension for binary
            else:
                pred = torch.softmax(outs[0], dim=1)  # (B, C, H, W)
                pred = torch.argmax(pred, dim=1)  # (B, H, W)
                target = outs[1]  # Already (B, H, W)

            self.save_overlay_images(input, pred, target, "test_"+str(i))

        preds = torch.cat(preds)
        targets = torch.cat(targets)

        test_loss = self.criterion(preds, targets)

        # MULTICLASS: Compute metrics using per-class scores
        if self.num_classes == 1:
            activation = 'sigmoid'
            test_dsc = smp_utils.metrics.Fscore(activation=activation)(preds.cpu(), targets.cpu())
        else:
            # For multiclass, compute per-class Dice and average (excluding background)
            # Convert predictions to probabilities
            pred_probs = torch.softmax(preds, dim=1)  # (N, C, H, W)
            # Convert targets to one-hot encoding
            targets_onehot = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()  # (N, C, H, W)

            # Compute Dice per class
            class_dsc = []
            for c in range(1, self.num_classes):  # Skip background (class 0)
                tp = torch.sum(pred_probs[:, c] * targets_onehot[:, c])
                fp = torch.sum(pred_probs[:, c] * (1 - targets_onehot[:, c]))
                fn = torch.sum((1 - pred_probs[:, c]) * targets_onehot[:, c])
                dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
                class_dsc.append(dice)
            test_dsc = torch.mean(torch.stack(class_dsc))
        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_dsc', test_dsc, on_step=False, on_epoch=True, prog_bar=True)

        self.test_metrics = {'test_loss': test_loss, 'test_dsc': test_dsc}

        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        trainable_params = (
            param for name, param in self.named_parameters()
            if not name.startswith('model.transformer')
        )
        for name, param in self.named_parameters():
            if name.startswith('model.transformer'):
                param.requires_grad = False
        return torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=0.01)

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
