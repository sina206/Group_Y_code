import os
import cv2
import numpy as np
from scipy.spatial import cKDTree
import random 

# helper functions for C3

# improve contrast of image with clahe 
def enhance_image(img):
    clahe = cv2.createCLAHE(clipLimit=4.0)
    enhanced_img = clahe.apply(img)
    return enhanced_img

# create gaussian pyramid 
def compute_gauss_pyramids(pyramid_levels, img):

    gp = [img]
    current_img = img

    for i in range(pyramid_levels - 1):
        current_img = cv2.pyrDown(current_img)
        gp.append(current_img)

    return gp

def extract_icon_features(icon_images, icon_dataset, pyramid_levels):
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.04, edgeThreshold=10)
    icon_features = {}

    for icon in icon_images:
        img = cv2.imread(os.path.join(icon_dataset, icon))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced_img = enhance_image(gray_img)
        img_pyramids = compute_gauss_pyramids(pyramid_levels, enhanced_img)

        features_per_level = []

        for level in img_pyramids:
            kp, desc = sift.detectAndCompute(level, None)
            features_per_level.append({"keypoints": kp, "descriptors": desc, "shape": level.shape[:2]})
        
        icon_features[icon] = features_per_level

    return icon_features

def extract_test_features(test_image, test_dataset, upscale_factor):
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.04, edgeThreshold=10)

    img = cv2.imread(os.path.join(test_dataset, test_image))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # upscale test images to match scale of icons in iconDataset
    upscaled_image = cv2.resize(gray_img, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)
    enhanced_img = enhance_image(upscaled_image)
    kp_sift, des_sift = sift.detectAndCompute(enhanced_img, None)
    test_features = {"keypoints": kp_sift, "descriptors": des_sift}

    return test_features

# compute raw noisy matches using BF method and kd trees
def match_feature(icon_img_desc, test_img_desc, max_ratio):

    raw_matches = []

    # check for sufficient features to perform matching on 
    if test_img_desc is None or len(test_img_desc) < 2 or icon_img_desc is None or len(icon_img_desc) < 2:
        return raw_matches
    
    tree = cKDTree(test_img_desc.astype(np.float64))

    sorted_distances, sorted_indices = tree.query(icon_img_desc.astype(np.float64), k=2)

    # extract top 2 distances
    dist_1 = sorted_distances[:, 0]
    dist_2 = sorted_distances[:, 1]

    # calculate lowe's ratio = d1/d2
    # ensure denominator is greater than 0 to prevent errors
    lowe_ratio = dist_1 / (dist_2 + 1e-8) 

    # mark valid matches with bool mask
    lowes_mask = lowe_ratio < max_ratio

    # apply lowes mask on test, icon and distances
    icon_des_index = np.where(lowes_mask)[0]        
    test_des_index = sorted_indices[lowes_mask, 0]
    valid_distances = dist_1[lowes_mask]   

    for x, y, z in zip(icon_des_index, test_des_index, valid_distances):
        raw_matches.append((x, y, z))

    return raw_matches

def compute_homography(icon_points, test_points):
   
    equation_matrix = []

    for i in range(len(icon_points)):
        x, y = icon_points[i]
        u, v = test_points[i]

        # build equations
        equation_matrix.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        equation_matrix.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    
    equation_matrix = np.array(equation_matrix)

    try:
        # solve equations
        U, s, Vh = np.linalg.svd(equation_matrix)
        params = Vh[-1]
        h = params.reshape(3,3) 
        h /= h[2, 2]
        return h
    
    # error handling 
    except np.linalg.LinAlgError:
        return None

def ransac(icon_points, test_points, inlier_threshold, max_iterations):

    if len(icon_points) < 4:
        return None
    
    inlier_indices = []
    best_model = None
    best_inlier_count = 0

    for i in range(max_iterations):
        # randomly sample 4 unique matches 
        random_match_indices = np.random.choice(len(icon_points),4, replace = False) 

        ransac_icon_points = icon_points[random_match_indices]
        ransac_test_points = test_points[random_match_indices]

        h = compute_homography(ransac_icon_points, ransac_test_points)

        # check for when homography not computed
        if h is None:
            continue

        projected_points = []
        valid = []


        for i in range(len(icon_points)):
            x = icon_points[i][0]
            y = icon_points[i][1]

            col = np.array([x, y, 1])  

            # project icon with homography
            p_point = h @ col 

            if abs(p_point[2]) < 1e-10:
                continue

            # convert points back to 2D
            projected_points.append([p_point[0]/p_point[2], p_point[1]/p_point[2]]) 
            valid.append(i) 

        projected_points = np.array(projected_points)

        # checks if no valid projected points were found
        if not valid:
            continue


        sq_diffs = (projected_points - test_points[valid])**2
        errors = np.sqrt(sq_diffs.sum(1))
        inliers = errors < inlier_threshold
        inliers_count = np.sum(inliers)

        if inliers_count > best_inlier_count:
            inlier_indices = np.array(valid)[np.where(inliers)[0]]
            best_inlier_count = inliers_count
            best_model = h

    # refit homography to all inliers
    if best_inlier_count >= 4:
        best_model = compute_homography(icon_points[inlier_indices], test_points[inlier_indices])
    
    return best_model, inlier_indices
  
def compute_bbox(homography, icon_h, icon_w, upscale_factor):

    top_left     = (0, 0)
    top_right    = (icon_w, 0)
    bottom_left  = (0, icon_h)
    bottom_right = (icon_w, icon_h)

    x_coords = []
    y_coords = []

    for point in [top_left, top_right, bottom_left, bottom_right]:
        x, y = point
        projected_point = homography @ np.array([x, y, 1])
        projected_x = projected_point[0] / projected_point[2]
        projected_y = projected_point[1] / projected_point[2]
        x_coords.append(projected_x)
        y_coords.append(projected_y)
    
    left   = min(x_coords) / upscale_factor
    top    = min(y_coords) / upscale_factor
    right  = max(x_coords) / upscale_factor
    bottom = max(y_coords) / upscale_factor

    bbox = [left, top, right, bottom]

    return bbox

def refine_bbox(bbox, test_img):

    padding = 3
    white_threshold = 230
   
    left, top, right, bottom = bbox

    left = int(left) 
    top = int(top) 
    right = int(right) 
    bottom = int(bottom) 

    # expand bbox
    exp_left = left - 8
    exp_top = top - 8
    exp_right = right + 8
    exp_bottom = bottom + 8

    height = test_img.shape[0]
    width = test_img.shape[1]
    
    large_l = max(0, exp_left)
    large_t = max(0, exp_top)
    large_r = min(width, exp_right)
    large_b = min(height, exp_bottom)
    
    if large_r <= large_l or large_b <= large_t or test_img[large_t:large_b, large_l:large_r].size == 0:
        return (left, top, right, bottom)

    kernel = np.ones((3, 3))


    content = test_img[large_t:large_b, large_l:large_r]

    # create a bool mask to find content and use morphology to denoise
    mask = cv2.morphologyEx(((content > 30) & (content < white_threshold)).astype(np.uint8),cv2.MORPH_OPEN, kernel)
    
    row_content = np.any(mask,1)
    row_indices = np.where(row_content)[0]

    col_content = np.any(mask,0)
    col_indices = np.where(col_content)[0]
    
    # shrink bbox to content 
    shrink_top    = large_t + row_indices[0]
    shrink_bottom = large_t + row_indices[-1] + 1
    shrink_left   = large_l + col_indices[0]
    shrink_right  = large_l + col_indices[-1] + 1

    shrink_left   = max(0, shrink_left - padding)
    shrink_top    = max(0, shrink_top - padding)
    shrink_right  = min(width, shrink_right + padding)
    shrink_bottom = min(height, shrink_bottom + padding)

    if len(row_indices) == 0 or len(col_indices) == 0:
        return (left, top, right, bottom)
    
    return (shrink_left, shrink_top, shrink_right, shrink_bottom)

def calculate_iou(real_box, predicted_box):

    a1, a2, b1, b2 = real_box
    c1, c2, d1, d2 = predicted_box

    if min(b1, d1) <= max(a1, c1) or min(b2, d2) <= max(a2, c2):
        return 0

    box_inter = (min(b1, d1) - max(a1, c1)) * (min(b2, d2) - max(a2, c2))
    box_x = (b1 - a1) * (b2 - a2)
    box_y = (d1 - c1) * (d2 - c2)
    box_union = box_x + box_y - box_inter

    if box_union <= 0:
        return 0

    iou = box_inter / box_union

    return iou

def compute_metrics(total_TP, total_FP, total_FN, runtime, test_set_size, all_ious):

    total = total_TP + total_FP + total_FN

    if total != 0:
        acc = total_TP / total

    if (total_TP + total_FN) != 0:
        tpr = total_TP / (total_TP + total_FN)
    
    if (total_TP + total_FP) != 0:
        fpr = total_FP / (total_TP + total_FP)
    
    if (total_TP + total_FN) != 0:
        fnr = total_FN / (total_TP + total_FN)

    avg_runtime = runtime / test_set_size

    if (total_TP + total_FP) != 0:
        precision = total_TP / (total_TP + total_FP)

    recall = tpr

    if (precision + recall) != 0:
        f1 = 2 * precision * recall / (precision + recall)

    mean_iou = (sum(all_ious) / len(all_ious)) 
    
    print("------------------------------------")
    print("FINAL METRICS:")
    print(f"true positives = {total_TP}")
    print(f"false positives = {total_FP}")
    print(f"false negatives = {total_FN}")
    print(f"accuracy = {acc:.3f}")
    print(f"TPR = {tpr:.3f}")
    print(f"FPR = {fpr:.3f}")
    print(f"FNR = {fnr:.3f}")
    print(f"Avg runtime per image = {avg_runtime:.3f}")
    print(f"Precision = {precision:.3f}")
    print(f"F1 = {f1:.3f}")
    print(f"Mean iou = {mean_iou:.3f}")

    return acc, tpr, fpr, fnr

def get_ground_truth(annotations_dataset, test_img):
    gt_file = test_img.replace(".png", ".csv")
    gt_path = os.path.join(annotations_dataset, gt_file)

    ground_truth_classes = []
    ground_truth_bboxes = []

    with open(gt_path, "r") as f:
        rows = f.readlines()

    for row in rows[1:]:  

        data = row.strip()
        data = data.split(",")

        icon_class = data[0].strip()

        left = int(data[1])
        top = int(data[2])
        right = int(data[3])
        bottom = int(data[4])


        ground_truth_classes.append(icon_class)
        ground_truth_bboxes.append(( left, top, right, bottom))
    
    return ground_truth_classes, ground_truth_bboxes

def draw_bbox(bbox, colour, image, classname=None):

    left_top = (int(bbox[0]), int(bbox[1]))
    right_bottom = (int(bbox[2]), int(bbox[3]))

    cv2.rectangle(image, left_top, right_bottom, colour, thickness=2)

    if classname is not None:
        cv2.putText(image, classname, (left_top[0], left_top[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

