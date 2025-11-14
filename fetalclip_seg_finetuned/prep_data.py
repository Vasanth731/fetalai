'''
Copy data in the form of root/images and root/masks into ./data folder, additionally rename the files in the format of <num>_<addendum>.png
and <num>_<addendum>_Annotation.png respectively.
Create test_id.json and train_val_split.json.
'''
import json
import os
import shutil
import random

data_root = "../../../../data/in-house/mediscan-seg/AC/images"
out_dir = "./data"
addendum = 'AC'

if os.path.exists(out_dir) and os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

imgs = os.listdir(data_root)
n = len(imgs)

# rename and copy from mediscan images folder
for i,img in enumerate(imgs):
    orig_path = os.path.join(data_root, img)
    orig_mask_path = os.path.join(data_root.replace('images', 'masks'), img)
    new_img = str(i)+'_'+addendum+'.png'
    new_mask_img = str(i)+'_'+addendum+'_Annotation'+'.png'
    new_path = os.path.join(out_dir, new_img)
    new_mask_path = os.path.join(out_dir, new_mask_img)
    print(i, img, new_img)
    shutil.copy2(orig_path, new_path)
    shutil.copy2(orig_mask_path, new_mask_path)
    
data = [str(i) for i in range(n)]
random.shuffle(data)
# Compute split indices
n = len(data)
train_end = int(n * 0.9)

# Split
train = data[:train_end]
test = data[train_end:]

# Assuming you already have a 'train' list
def split_into_chunks(lst, num_chunks):
    k, m = divmod(len(lst), num_chunks)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(num_chunks)]

train_parts = split_into_chunks(train, 5)

train_val = {}
for i, lst in enumerate(train_parts):
    train_end = int(len(lst) * 0.9)
    train_ = lst[:train_end]
    val_ = lst[train_end:]
    train_val[str(i)] = [train_, val_]

with open("train_val_split.json", "w") as f:
    json.dump(train_val, f, indent=4) 

test_id = {}
for i in test:
    test_id[str(i)] = "test"
with open("test_id.json", "w") as f:
    json.dump(test_id, f, indent=4) 
