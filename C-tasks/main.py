import argparse
import pandas as pd
import time
from task_c1_utils import *
from task_c3_utils import *
from task_c2_utils import *


def test_task_c1(folder_name):
    predAngles, actual_angles = [], []

    # assume that this folder name has a file list.txt that contains the annotation.
    task1_data = pd.read_csv(folder_name + "/list.txt")

    for idx, row in task1_data.iterrows():
        # Write code to read in each image
        img_path = os.path.join(folder_name, row["FileName"])
        actual_angle = row["AngleInDegrees"]
        actual_angles.append(actual_angle)

        img = cv2.imread(img_path)

        # Write code to process the image
        processed_img = process_img(img)

        # Apply canny edge detection to help us find the lines
        edges = canny_edge_detection(processed_img)

        # Apply hough transform on detected edges to find lines (note: these are not our final lines)
        accumulator, rhos, thetas = hough_transform(edges)
        peaks = find_hough_peaks(accumulator, thetas, rhos, num_peaks=NUM_PEAKS, nhood_size=NHOOD_SIZE)

        # Invalid image or insufficient peaks detected
        if len(peaks) < 2:
            # Remove the angle we added for this failed image
            actual_angles.pop()
            # Move to the next image
            continue

        # Write your code to calculate the angle and obtain the result as a list predAngles
        estimated_angle = interior_angle_from_edges(peaks, edges, 8)
        predAngles.append(estimated_angle)

    # Calculate and provide the error in predicting the angle for each image
    # We use mean absolute error
    total_error = 0
    for pred, actual in zip(predAngles, actual_angles):
        total_error += abs(pred - actual)
    total_error /= len(predAngles)
    return total_error


def test_task_c2(icon_dir, test_dir):
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives
    start_time = time.time()
    
    results_dir = create_results_folder("c2_results")
    print("Saving outputs to:", results_dir)
    
    TP_total = 0
    FP_total = 0
    FN_total = 0
    total_iou = 0.0
    num_iou_matches = 0
    
    N_JOBS = -1 ## Maximises parallel execution ***(adjust based on hardware capacity!)***
        
    ## Here we preprocess the test images and icons
    icon_image_arr = []
    for folder in os.listdir(icon_dir):
        if folder=="png":
            for icon in os.listdir(os.path.join(icon_dir, folder)):
                img_path = os.path.join(icon_dir, folder, icon)
                grey_icon = cv2.imread(img_path, 0)
                grey_icon[grey_icon > 240] = 0
                icon_image_arr.append([icon, grey_icon])
    
    sorted_icon_image_arr = natsorted(icon_image_arr)
    
    #sort test images so we get folders in order: annotations,images
    test_images_folders = sorted(os.listdir(test_dir)) 
    sorted_annotations_files = natsorted(os.listdir(os.path.join(test_dir, test_images_folders[0])))
    sorted_images_files = natsorted(os.listdir(os.path.join(test_dir, test_images_folders[1])))
    
    test_image_arr = []
    for annotations, images in zip(sorted_annotations_files, sorted_images_files):
        grey_test_image = cv2.imread(os.path.join(test_dir,test_images_folders[1],images),0)
        colour_test_image = cv2.imread(os.path.join(test_dir,test_images_folders[1],images))
        grey_test_image[grey_test_image > 240] = 0
        test_image_arr.append([annotations,grey_test_image,colour_test_image])

    print("Icon and test datasets loaded")
    
    ## Here we create Gaussian Pyramids for each icon
    pyramids = {}
    for icon_name, icon_img in sorted_icon_image_arr:
        icon_pyramid = build_normalised_g_Pyramids(icon_img)
        pyramids[icon_name] = icon_pyramid
        
    p_icon_name, p_icon_img = sorted_icon_image_arr[0]
    pyramid_scales = [pyramid.shape for pyramid in pyramids[p_icon_name]]

    print("Finished creating Gaussian pyramids")
    
    print("Starting template matching")
    
    ## Here we set-up parallelisation and perform template matching over test images
    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(process_single_test_image)(
            test_image_csv,
            test_image,
            colour_test_image,
            sorted_icon_image_arr,
            pyramids,
            test_dir,
            test_images_folders,
            results_dir
        )
        for test_image_csv, test_image, colour_test_image in test_image_arr
    )
    
    ## Here we calculate final metrics across all test images
    for TP, FP, FN, img_iou, img_iou_count in results:
        TP_total += TP
        FP_total += FP
        FN_total += FN
        total_iou += img_iou
        num_iou_matches += img_iou_count

    end_time = time.time()
    runtime = end_time - start_time
    avg_runtime_per_image = (runtime) / len(test_image_arr)
    print("Average runtime per image:", avg_runtime_per_image)
    
    avg_iou = total_iou / (num_iou_matches + 1e-8)
    print("Average IoU:", avg_iou)
    
    eps = 1e-8

    tpr = TP_total / (TP_total + FN_total + eps)
    fnr = FN_total / (TP_total + FN_total + eps)
    fpr = FP_total / (FP_total + TP_total + eps)
    acc = TP_total / (TP_total + FP_total + FN_total + eps)
    
    precision = TP_total / (TP_total + FP_total + eps)
    f1 = 2 * precision * tpr / (precision + tpr + eps)
    
    print("ACC:", acc)
    print("TPR:", tpr)
    print("FPR:", fpr)
    print("FNR:", fnr)
    print("Precision:", precision)
    print("F1 Score:", f1)
    
    ## Here we save final metrics to the results folder
    results_path = os.path.join(results_dir, "metrics.csv")

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "test_images_folder",
            "ncc_threshold",
            "pyramid_scales",
            "iou",
            "avg_runtime_per_image_sec",
            "runtime",
            "avg_iou",
            "precision",                   
            "f1",
            "accuracy",
            "tpr",
            "fpr",
            "fnr",
            "step size",
            "parallel_n_jobs" 
        ])
        writer.writerow([
            results_dir,
            NCC_THRESHOLD,
            pyramid_scales,
            IOU_NMS,
            avg_runtime_per_image,
            runtime,
            avg_iou,
            precision,                   
            f1,
            acc,
            tpr,
            fpr,
            fnr,
            STEP,
            N_JOBS
        ])
    
    return acc, tpr, fpr, fnr


def test_task_c3(icon_folder_name, test_folder_name):
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives

    # load images and annotations
    icon_dataset = os.path.join(args.IconDataset, "png")
    test_dataset = os.path.join(args.Task3Dataset, "images")
    annotations_dataset = os.path.join(args.Task3Dataset, "annotations")
    icon_images = os.listdir(icon_dataset)
    test_images = os.listdir(test_dataset)


    print("icons and test datasets loaded!")

    # CONFIG
    upscale_factor = 5 
    pyramid_levels = 4
    max_ratio = 0.70   # threshold for lowe's ratio 
    inlier_threshold = 5
    max_iterations = 500 # ransac iterations
    min_inliers_count = 6

    # METRICS
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    np.random.seed(42)

    # extract sift feature from icons 
    icon_features = extract_icon_features(icon_images, icon_dataset, pyramid_levels)
    
    print("sift features extracted!")

    all_ious = []
    time_taken = 0.0

    # set results path to task3dataset
    results_dir = os.path.join(args.Task3Dataset, "c3_results")
    os.makedirs(results_dir, exist_ok=True)

    for test_img in test_images:
        t0 = time.time()

        # extract test image keypoints and descriptors
        test_features = extract_test_features(test_img, test_dataset, upscale_factor)
        test_desc = test_features["descriptors"]

        # save a coloured version of test for bbox plotting
        test_colour = cv2.imread(os.path.join(test_dataset, test_img))
        test = cv2.cvtColor(test_colour, cv2.COLOR_BGR2GRAY)

        gt_classes, gt_bboxes = get_ground_truth(annotations_dataset, test_img)

        correct_matches = set()

        # iterate through all icon features 
        for icon in icon_features:

            best_model = None
            best_inliers_count = 0
            best_gp_level = None

            # iterate through all levels of pyramid
            for icon_gp_level in range(len(icon_features[icon])):


                # extract descriptors of icon at that gp level
                icon_desc = icon_features[icon][icon_gp_level]["descriptors"]
                if icon_desc is None or len(icon_desc) < 2:
                    continue

                # generate raw matches with that level icon descriptors
                raw_matches = match_feature(icon_desc, test_desc, max_ratio)
                if len(raw_matches) < 4:
                    continue

                icon_keypoints = icon_features[icon][icon_gp_level]["keypoints"]
                test_keypoints = test_features["keypoints"]

                # extract keypoints of all test and icon matches
                icon_points = []
                test_points = []

                for i in raw_matches:
                    icon_points.append(icon_keypoints[i[0]].pt)
                    test_points.append(test_keypoints[i[1]].pt)
                
                icon_points = np.array(icon_points)
                test_points = np.array(test_points)

                # run ransac on raw matches
                local_model, inlier_indices = ransac(icon_points, test_points, inlier_threshold, max_iterations)

                local_inliers_count = len(inlier_indices)

                # update best models
                if local_model is not None and local_inliers_count > best_inliers_count:
                    best_inliers_count = len(inlier_indices)
                    best_model = local_model
                    best_gp_level = icon_gp_level
            
            # accept model if inliers are above the minimum
            if best_model is not None and best_inliers_count >= min_inliers_count:

                icon_h, icon_w = icon_features[icon][best_gp_level]["shape"]

                corners = np.array([[0,0,1],[icon_w,0,1],[icon_w,icon_h,1],[0,icon_h,1]])
                corners = corners.T

                # project best model from icon -> test image
                projected = best_model @ corners
                w = projected[2, :]
                
                points = (projected[:2, :] / w)
                points = points.T
                
                # check for non-convex model by checking all cross products have the same sign
                signs = []

                for i in range(len(points)):

                    cross_prod_1 = (points[(i + 1) % 4][0] - points[i][0]) * (points[(i + 2) % 4][1] - points[(i + 1) % 4][1])
                    cross_prod_2 = (points[(i + 1) % 4][1] - points[i][1]) * (points[(i + 2) % 4][0] - points[(i + 1) % 4][0])

                    diff = cross_prod_1 - cross_prod_2

                    if diff < 0:
                        signs.append("-")
                    else:
                        signs.append("+")

                if len(set(signs)) != 1:
                    continue
                
                # using model compute bounding box corners
                bounding_box = compute_bbox(best_model, icon_h, icon_w, upscale_factor)

                # eliminate irregularly sized bounding boxes
                left, top, right, bottom = bounding_box

                width = right - left
                height = bottom - top

                if height < 3 or height > 350:
                    continue
                if width < 3 or width > 350:
                    continue

                # shrink bounding box to real content
                refined_bbox = refine_bbox(bounding_box, test)  

                # get name of predicted class
                predicted_icon_class = icon.replace(".png", "")
                predicted_icon_class = predicted_icon_class.split("-", 1)[1]
                
                best_iou = 0.0
                best_gt_i = -1
            
                # find the correct ground truth
                for i in range(len(gt_classes)):

                    gt = gt_classes[i]

                    if predicted_icon_class == gt:

                        iou = calculate_iou(refined_bbox, gt_bboxes[i])

                        if iou > best_iou:

                            best_iou = iou
                            best_gt_i = i
                
                all_ious.append(best_iou)
                
                # check whether predictions are correct and plot bboxes
                if best_iou >= 0.85 and best_gt_i != -1 and best_gt_i not in correct_matches:
                    true_positives += 1
                    correct_matches.add(best_gt_i)
                    draw_bbox(refined_bbox, (255,0, 0), test_colour, icon)
                else:
                    false_positives += 1
                    draw_bbox(refined_bbox, (0, 0, 255), test_colour, icon)
                    

        # count up false negatives and plot these missed ground truths
        for i in range(len(gt_classes)):

            gt = gt_classes[i]
            gt_bbox = gt_bboxes[i]

            if i not in correct_matches: 
                # draw gt if FN
                draw_bbox(gt_bbox, (0, 255, 0), test_colour, None)

        false_negatives += len(gt_classes) - len(correct_matches)
        cv2.imwrite(os.path.join(results_dir, test_img), test_colour)


        print("------------------------------------")
        print("finished processing: " + test_img)
        print(f"no of true matches: {len(correct_matches)}")

        time_taken += (time.time() - t0)

    test_set_size = len(test_images)

    acc, tpr, fpr, fnr = compute_metrics(true_positives, false_positives, false_negatives, time_taken, test_set_size, all_ious)

    return acc, tpr, fpr, fnr


if __name__ == "__main__":
    # parsing the command line path to directories and invoking the test scripts for each task
    parser = argparse.ArgumentParser("Data Parser")
    parser.add_argument(
        "--Task1Dataset", help="Provide a folder that contains the Task 1 Dataset.", type=str, required=False
    )
    parser.add_argument(
        "--IconDataset",
        help="Provide a folder that contains the Icon Dataset for Task2 and Task3.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--Task2Dataset", help="Provide a folder that contains the Task 2 test Dataset.", type=str, required=False
    )
    parser.add_argument(
        "--Task3Dataset", help="Provide a folder that contains the Task 3 test Dataset.", type=str, required=False
    )
    args = parser.parse_args()

    if args.Task1Dataset is not None:
        # This dataset has a list of png files and a txt file that has annotations of filenames and angle
        test_task_c1(args.Task1Dataset)

    if args.IconDataset is not None and args.Task2Dataset is not None:
        # The Icon dataset has a directory that contains the icon image for each file
        # The Task2 dataset directory has two directories, an annotation directory that contains the annotation and a
        # png directory with list of images
        test_task_c2(args.IconDataset, args.Task2Dataset)

    if args.IconDataset is not None and args.Task3Dataset is not None:
        # The Icon dataset directory contains an icon image for each file
        # The Task3 dataset has two directories, an annotation directory that contains the annotation and a png
        # directory with list of images
        test_task_c3(args.IconDataset, args.Task3Dataset)
