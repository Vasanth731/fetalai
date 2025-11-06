import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import argparse
from pathlib import Path

from model import YNet
from dataloader import get_dataloaders


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] - predicted mask (after sigmoid)
            target: [B, 1, H, W] - ground truth mask
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined loss for classification and segmentation"""
    def __init__(self, seg_weight=1.0, cls_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, class_output, seg_output, labels, masks):
        """
        Args:
            class_output: [B, 4] - classification logits
            seg_output: [B, 1, H, W] - segmentation output (after sigmoid)
            labels: [B] - class labels (1, 2, or 3)
            masks: [B, 1, H, W] - ground truth masks
        """
        # Segmentation losses
        dice_loss = self.dice_loss(seg_output, masks)
        bce_loss = self.bce_loss(seg_output, masks)
        seg_loss = dice_loss + bce_loss

        # Classification loss
        cls_loss = self.ce_loss(class_output, labels)

        # Combined loss
        total_loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss

        return total_loss, seg_loss, cls_loss


def freeze_encoder(model):
    """
    Freeze the encoder part of the model.
    Encoder includes: down_conv1-4, max_pool_2x2, conv_lstm, bootleneck
    """
    # Freeze downsampling convolutions
    for param in model.down_conv1.parameters():
        param.requires_grad = False
    for param in model.down_conv2.parameters():
        param.requires_grad = False
    for param in model.down_conv3.parameters():
        param.requires_grad = False
    for param in model.down_conv4.parameters():
        param.requires_grad = False

    # Freeze ConvLSTM
    for param in model.conv_lstm.parameters():
        param.requires_grad = False

    # Freeze bottleneck
    for param in model.bootleneck.parameters():
        param.requires_grad = False

    print("Encoder frozen successfully!")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")


def calculate_metrics(class_output, seg_output, labels, masks, threshold=0.5):
    """Calculate accuracy and IoU metrics"""
    # Classification accuracy
    _, predicted_classes = torch.max(class_output, 1)
    cls_correct = (predicted_classes == labels).sum().item()
    cls_accuracy = cls_correct / labels.size(0)

    # Segmentation IoU
    pred_masks = (seg_output > threshold).float()
    intersection = (pred_masks * masks).sum().item()
    union = pred_masks.sum().item() + masks.sum().item() - intersection
    iou = intersection / (union + 1e-8)

    return cls_accuracy, iou


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    # Keep encoder in eval mode to prevent batch norm updates
    model.down_conv1.eval()
    model.down_conv2.eval()
    model.down_conv3.eval()
    model.down_conv4.eval()
    model.conv_lstm.eval()
    model.bootleneck.eval()

    running_loss = 0.0
    running_seg_loss = 0.0
    running_cls_loss = 0.0
    running_cls_acc = 0.0
    running_iou = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for images, masks, labels in pbar:
        images = images.to(device)  # [B, 1, 1, H, W]
        masks = masks.to(device)    # [B, 1, H, W]
        labels = labels.to(device)  # [B]

        # Forward pass
        optimizer.zero_grad()
        class_output, seg_output = model(images)

        # Calculate loss
        total_loss, seg_loss, cls_loss = criterion(class_output, seg_output, labels, masks)

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Calculate metrics
        cls_acc, iou = calculate_metrics(class_output, seg_output, labels, masks)

        # Update running metrics
        running_loss += total_loss.item()
        running_seg_loss += seg_loss.item()
        running_cls_loss += cls_loss.item()
        running_cls_acc += cls_acc
        running_iou += iou

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'cls_acc': f'{cls_acc:.4f}',
            'iou': f'{iou:.4f}'
        })

    # Calculate epoch averages
    num_batches = len(train_loader)
    avg_loss = running_loss / num_batches
    avg_seg_loss = running_seg_loss / num_batches
    avg_cls_loss = running_cls_loss / num_batches
    avg_cls_acc = running_cls_acc / num_batches
    avg_iou = running_iou / num_batches

    return avg_loss, avg_seg_loss, avg_cls_loss, avg_cls_acc, avg_iou


def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()

    running_loss = 0.0
    running_seg_loss = 0.0
    running_cls_loss = 0.0
    running_cls_acc = 0.0
    running_iou = 0.0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    with torch.no_grad():
        for images, masks, labels in pbar:
            images = images.to(device)  # [B, 1, 1, H, W]
            masks = masks.to(device)    # [B, 1, H, W]
            labels = labels.to(device)  # [B]

            # Forward pass
            class_output, seg_output = model(images)

            # Calculate loss
            total_loss, seg_loss, cls_loss = criterion(class_output, seg_output, labels, masks)

            # Calculate metrics
            cls_acc, iou = calculate_metrics(class_output, seg_output, labels, masks)

            # Update running metrics
            running_loss += total_loss.item()
            running_seg_loss += seg_loss.item()
            running_cls_loss += cls_loss.item()
            running_cls_acc += cls_acc
            running_iou += iou

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'cls_acc': f'{cls_acc:.4f}',
                'iou': f'{iou:.4f}'
            })

    # Calculate epoch averages
    num_batches = len(val_loader)
    avg_loss = running_loss / num_batches
    avg_seg_loss = running_seg_loss / num_batches
    avg_cls_loss = running_cls_loss / num_batches
    avg_cls_acc = running_cls_acc / num_batches
    avg_iou = running_iou / num_batches

    return avg_loss, avg_seg_loss, avg_cls_loss, avg_cls_acc, avg_iou


def train(args):
    """Main training function"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create tensorboard writer
    writer = SummaryWriter(log_dir=output_dir / "logs")

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        root_folder=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(224, 224)
    )

    # Create model
    print("Creating model...")
    model = YNet(
        input_channels=1,
        output_channels=64,
        n_class=1
    )

    # Load pretrained weights
    if args.pretrained_weights:
        print(f"Loading pretrained weights from {args.pretrained_weights}")
        state_dict = torch.load(args.pretrained_weights, map_location=device)
        model.load_state_dict(state_dict)
        print("Pretrained weights loaded successfully!")

    # Freeze encoder
    freeze_encoder(model)

    # Move model to device
    model = model.to(device)

    # Create loss function and optimizer
    criterion = CombinedLoss(seg_weight=args.seg_weight, cls_weight=args.cls_weight)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Training loop
    best_val_loss = float('inf')
    print("\nStarting training...")
    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_loss, train_seg_loss, train_cls_loss, train_cls_acc, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_seg_loss, val_cls_loss, val_cls_acc, val_iou = validate_epoch(
            model, val_loader, criterion, device, epoch
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('SegLoss/train', train_seg_loss, epoch)
        writer.add_scalar('SegLoss/val', val_seg_loss, epoch)
        writer.add_scalar('ClsLoss/train', train_cls_loss, epoch)
        writer.add_scalar('ClsLoss/val', val_cls_loss, epoch)
        writer.add_scalar('ClsAcc/train', train_cls_acc, epoch)
        writer.add_scalar('ClsAcc/val', val_cls_acc, epoch)
        writer.add_scalar('IoU/train', train_iou, epoch)
        writer.add_scalar('IoU/val', val_iou, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Seg: {train_seg_loss:.4f}, Cls: {train_cls_loss:.4f}, "
              f"Acc: {train_cls_acc:.4f}, IoU: {train_iou:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Seg: {val_seg_loss:.4f}, Cls: {val_cls_loss:.4f}, "
              f"Acc: {val_cls_acc:.4f}, IoU: {val_iou:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved! (val_loss: {val_loss:.4f})")

    # Save final model
    final_model_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining complete! Final model saved to {final_model_path}")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune YNet for fetal biometry")

    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to root folder containing BPD, AC, FL subfolders')
    parser.add_argument('--pretrained_weights', type=str, required=True,
                        help='Path to pretrained weights file')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer (default: 1e-5)')

    # Loss weights
    parser.add_argument('--seg_weight', type=float, default=1.0,
                        help='Weight for segmentation loss (default: 1.0)')
    parser.add_argument('--cls_weight', type=float, default=0.5,
                        help='Weight for classification loss (default: 0.5)')

    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save checkpoints and logs (default: ./output)')

    args = parser.parse_args()

    # Print configuration
    print("="*60)
    print("Training Configuration")
    print("="*60)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("="*60)

    # Start training
    train(args)
