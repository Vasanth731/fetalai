import os
import time
import cv2 as cv
import numpy as np
import csv

def detect_arrowhead(path, net, conf=0.5):
    # global img, img0, outputs, ln
    
    # determine the output layers
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    img0 = cv.imread(path)
    img = img0.copy()
    
    blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

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

    indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
    locs = []
    if len(indices)>0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            center = int(x + w//2), int(y + h//2)
            locs.append(center)

            cv.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
            text = "{}: {:.4f}".format("ArrowHead", confidences[i])
            cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    else:
        locs = None
    return img, locs


def detect_all(data_root, yolo_weight, yolo_cfg, num_points, dest="dataset/masks/AC", save_figs=False):
    imgs = os.listdir(data_root)
    if save_figs:
        os.makedirs(dest, exist_ok=True)
        os.makedirs(os.path.join(dest, "bad"), exist_ok=True)
        os.makedirs(os.path.join(dest, "good"), exist_ok=True)

    # Give the configuration and weight files for the model and load the network.
    net = cv.dnn.readNetFromDarknet(yolo_cfg, yolo_weight)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

    data = {}
    for x in imgs:
        path = os.path.join(data_root, x)
        img, locs = detect_arrowhead(path, net)
        
        if save_figs:
            if locs == None or len(locs) != num_points:
                cv.imwrite(os.path.join(dest, "bad", x), img)
            else:
                cv.imwrite(os.path.join(dest, "good", x), img)
        
        if locs is not None and len(locs)==num_points:
            data[x] = locs
    return data


def save_detections_to_csv(data, csv_filename="detections.csv"):
    """Saves detected bounding box coordinates along with filenames to a CSV file."""
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Coordinates"])
        
        for filename, coordinates in data.items():
            writer.writerow([filename, coordinates])

    
if __name__ == "__main__":
    # location of image folder
    data_inst = detect_all(
        data_root="/home/htic/vasanth/FetalCLIP/data2/5plusextracls_inhouse/inhouse_brain_transthalamic",  # dataset/images/AC
        yolo_weight="yolov4-tiny-arrowhead.weights",
        yolo_cfg="yolov4-tiny-arrowhead.cfg",
        num_points = 4,
        save_figs=True)
    
    print(f"{type(data_inst)}")
    
    save_detections_to_csv(data_inst, "dataset/masks/detections_AC.csv")

