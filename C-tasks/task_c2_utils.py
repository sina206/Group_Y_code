import argparse
import os
import pandas as pd
import cv2
from natsort import natsorted
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import csv
import time
from joblib import Parallel, delayed

CUSTOM_SCALES   = [0.625, 0.5, 0.375, 0.25, 0.125]
NCC_THRESHOLD   = 0.65
IOU_THRESHOLD   = 0.85
IOU_NMS         = 0.1
STEP            = 2

def create_results_folder(base_name="c2_results"):
        i = 1
        while True:
            folder_name = f"{base_name} ({i})"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                return folder_name
            i += 1

## Here we create Gaussian Pyramids for each icon
def build_normalised_g_Pyramids(img, scales=CUSTOM_SCALES):
    pyramid = []
    h, w = img.shape
        
    for scale in scales:
        resized_img = cv2.resize(img, (int(w*scale), int(h*scale)),interpolation=cv2.INTER_AREA)
        resized_img = resized_img.astype(np.float32)
        
        # normalise
        template_mean = np.mean(resized_img) 
        template_std = np.std(resized_img)
        normalised = (resized_img - template_mean) / (template_std + 1e-8)
        
        pyramid.append(normalised.astype(np.float32))
    
    return pyramid

## Similarity calculation function
def ZNCC(normalised_template, patch):
    patch = patch.astype(np.float32)
    patch_mean = patch.mean()
    patch_std = patch.std() + 1e-8
    norm_patch = (patch - patch_mean) / patch_std

    return (norm_patch * normalised_template).sum() / normalised_template.size

## Here we calculate IOU & Containment
def box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

def intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return max(0, xB - xA) * max(0, yB - yA)

def compute_iou(boxA, boxB):
    inter = intersection_area(boxA, boxB)
    union = box_area(boxA) + box_area(boxB) - inter
    return inter / union if union > 0 else 0

def compute_containment(boxA, boxB):
    inter = intersection_area(boxA, boxB)
    areaA = box_area(boxA)
    areaB = box_area(boxB)
    smaller_area = min(areaA, areaB)
    return inter / smaller_area if smaller_area > 0 else 0

## Non-Maximum Supression function
def nms(final_detections, iou_thresh=IOU_NMS):
    if len(final_detections) == 0:
        return []

    keep_detections = []

    for i, det_i in enumerate(final_detections):
        discard = False
        for j, det_j in enumerate(final_detections):
            if j == i:
                continue

            iou = compute_iou(det_i["bbox"], det_j["bbox"])
            containment = compute_containment(det_i["bbox"], det_j["bbox"])
            if (iou > iou_thresh or containment > 0.2) and det_j["score"] > det_i["score"]:
                discard = True
                break

        if not discard:
            keep_detections.append(det_i)

    return keep_detections
                    
def extract_ground_truths(csv_path):
    gt = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header row

        for row in reader:
            classname = row[0]
            top = int(row[1])
            left = int(row[2])
            bottom = int(row[3])
            right = int(row[4])

            gt.append({
                "class": "0" + classname + ".png",
                "bbox": [top, left, bottom, right]
            })
    return gt

## Here is where we perform template matching for a single test image
def process_single_test_image(
    test_image_csv,
    test_image,
    colour_test_image,
    sorted_icon_image_arr,
    pyramids,
    test_dir,
    test_images_folders,
    results_dir
):
    # set all metrics to 0 
    TP = FP = FN = 0
    total_iou = 0.0
    num_iou_matches = 0

    all_icon_detections = []

    print("processing test image: ", test_image_csv)
    for icon_name, _ in sorted_icon_image_arr: # across all icons
        for g_scale in pyramids[icon_name]: # across each gaussian scale
                            
            test_image_f = test_image.astype(np.float32)
            img_h, img_w = test_image_f.shape
            h, w = g_scale.shape

            detections = []

            for y in range(0, img_h - h + 1, STEP):
                for x in range(0, img_w - w + 1, STEP):
                    patch = test_image_f[y:y+h, x:x+w]

                    score = ZNCC(g_scale, patch)
                    
                    if score >= NCC_THRESHOLD:
                        detections.append({
                            "class": icon_name,
                            "bbox": [x, y, x + w, y + h],
                            "score": score
                        })
            
            #remove overlapping detections
            final_detections = nms(detections, iou_thresh=IOU_NMS) 
            all_icon_detections.extend(final_detections)
        
        #final check to remove overlapping detections
        all_icon_detections = nms(all_icon_detections, iou_thresh=IOU_NMS) 
            
    ## Here we extract the ground truth annotations     
    gt_boxes = extract_ground_truths(os.path.join(test_dir, test_images_folders[0], test_image_csv))
    
    ## Here we caluclate evalution metrics for the test image
    TP, FP = 0, 0
    matched_gt = set()

    for pred in all_icon_detections:
        best_iou = 0
        best_gt_idx = -1

        for i, gt in enumerate(gt_boxes):
            if gt["class"] != pred["class"]:
                continue

            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        pred["best_iou"] = best_iou

        if best_iou >= IOU_THRESHOLD and best_gt_idx not in matched_gt:
            TP += 1
            matched_gt.add(best_gt_idx)
            total_iou += best_iou
            num_iou_matches += 1
        else:
            FP += 1

    FN = len(gt_boxes) - len(matched_gt)
            
    ## Here we draw bounding boxes on the test image
    for gt in gt_boxes:
        x1, y1, x2, y2 = gt["bbox"]
        cv2.rectangle(colour_test_image, (x1,y1), (x2,y2), (0,255,0), 2)

    
    for det in all_icon_detections:
        x1,y1,x2,y2 = det["bbox"]
        iou = det.get("best_iou", 0)

        if iou >= IOU_THRESHOLD:
            color = (255,0,0)  # BLUE = TP
            label = f"{det['class']}"
        else:
            color = (0,0,255)  # RED = FP
            label = f"{det['class']}"

        cv2.rectangle(colour_test_image, (x1,y1), (x2,y2), color, 2)

        img_h = colour_test_image.shape[0]
        label_y = y1 - 5
        if label_y < 0:
            label_y = img_h - 10
        
        cv2.putText(
            colour_test_image,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )
    
    ## Here the test image with bounding boxes drawn is saved to the results folder
    out_path = os.path.join(results_dir, test_image_csv.replace(".csv", ".png"))
    cv2.imwrite(out_path, colour_test_image)

    return TP, FP, FN, total_iou, num_iou_matches