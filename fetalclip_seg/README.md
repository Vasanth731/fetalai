# Segmentation
This folder containes scripts used for probing Fetal-CLIP for segmentation

# How to run
1. Assuming your dataset is in the format of:
--><class>
    --> images
    --> masks

Run /python prep_data.py/ to move them into a folder named data along with renaming in the format of <num>_<class>.png and <num>_<class>_Annotation.png.

2. /python preprocess.py/
Generate embeddings using FetalCLIP's visual encoder and save image and mask in a single npz file.

3. Edit config.json and run /python train_fetalclip.py/ for beginning the training.

4. predict.py contains code for running the inference on a folder of images and a video clip.
