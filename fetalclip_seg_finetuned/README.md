# Segmentation
This folder contains scripts used for probing Fetal-CLIP for segmentation.

**Now supports both binary and multiclass segmentation!**

# Multiclass Segmentation Support

The codebase has been updated to support multiclass segmentation. The scripts automatically detect whether you're doing binary or multiclass segmentation based on `NUM_CLASSES`.

## Key Configuration

Before running, update the following in your scripts:

### 1. In `utils.py` (lines 52-58):
```python
LIST_CLASSES = ['head']  # UPDATE THIS: Add all your class names
                         # Example: ['background', 'head', 'body', 'limbs']
NUM_CLASSES  = len(LIST_CLASSES)  # Total number of output classes
```

### 2. In `predict.py` (lines 34-39):
```python
LIST_CLASSES = ['Femur']  # UPDATE THIS: Add all your class names
NUM_CLASSES  = len(LIST_CLASSES)
```

### 3. In `preprocess.py` (line 26):
```python
NUM_CLASSES = 1  # UPDATE THIS: Set to your number of classes
```

## Annotation Format for Multiclass

Your annotation images should have pixel values corresponding to class indices:
- **0**: Background (or first class)
- **1**: Class 1
- **2**: Class 2
- **3**: Class 3
- etc.

**Important**: Do NOT use binary masks (0 and 255). Use class indices (0, 1, 2, 3, ...).

## Color Mapping for Visualization

The scripts use the following colors for different classes:
- Class 0: Black (background)
- Class 1: Red
- Class 2: Green
- Class 3: Blue
- Class 4: Yellow
- Class 5: Magenta
- Class 6: Cyan
- Class 7: Orange
- Class 8: Purple
- Class 9: Teal

# How to Run

1. **Prepare your dataset** in the format:
   ```
   --><class>
       --> images
       --> masks
   ```
   Run `python prep_data.py` to organize them into a folder named `data` with the format:
   - `<num>_<class>.png` (images)
   - `<num>_<class>_Annotation.png` (masks with class indices)

2. **Update configurations** in `preprocess.py`, `utils.py`, and `predict.py`:
   - Set `NUM_CLASSES` to the number of classes
   - Set `LIST_CLASSES` to the list of class names

3. **Preprocess data**:
   ```bash
   python preprocess.py
   ```
   This generates embeddings using FetalCLIP's visual encoder and saves images and masks in npz files.

4. **Train the model**:
   Edit `config.json` and run:
   ```bash
   python train_fetalclip.py
   ```

5. **Run inference**:
   `predict.py` contains code for running inference on:
   - A folder of images
   - A video clip

   Update the checkpoint path and data directory in `predict.py` before running.

## Key Changes for Multiclass Support

The scripts now automatically handle:
- **Binary segmentation** (NUM_CLASSES=1): Uses sigmoid activation and binary cross-entropy loss
- **Multiclass segmentation** (NUM_CLASSES>1): Uses softmax activation and multiclass cross-entropy loss

Output masks will have different colored overlays for each class, making it easy to visualize multi-class predictions.
