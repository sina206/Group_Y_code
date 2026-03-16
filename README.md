# Project Submission
- We provide our project as a GitHub repository: https://github.com/sina206/Group_Y_code
- Moodle limits submissions to files under 50 MB and a maximum of five files. Using GitHub avoids splitting the repository across multiple ZIP files and reconstructing it.

# Contents

* [Dependency Installation Instructions](#dependency-installation-instructions)
* [Additional Setup Instructions](#additional-setup-instructions).
* [C-Task Execution](#c-task-execution)


# Dependency Installation Instructions

## 1. Create a Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv

# Windows
python -m venv venv
```

## 2. Activate the Virtual Environment
```bash
# macOS/Linux
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

## 3. Install Dependencies
All dependencies for our project are included the ```requirements.txt``` file:
```bash
pip install -r requirements.txt
```

# Additional Setup Instructions
**Note** - C2 and C3 assume the following folder structure:

```
C-tasks/
├── IconDataset/
│   └── png/
│       
├── Task2Dataset/
│   ├── annotations/
│   └── images/
│       
└── Task3Dataset/
    ├── annotations/
    └── images/
```     


## Loading Task C2 Dataset
Code (lines 72-94 of C-tasks/main.py):


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
    for annotations, images in zip(sorted_annotations_files[0:1], sorted_images_files[0:1]):
        grey_test_image = cv2.imread(os.path.join(test_dir,test_images_folders[1],images),0)
        colour_test_image = cv2.imread(os.path.join(test_dir,test_images_folders[1],images))
        grey_test_image[grey_test_image > 240] = 0
        test_image_arr.append([annotations,grey_test_image,colour_test_image])


## Loading Task C3 Dataset
Code (lines 210-214 of C-tasks/main.py):

    icon_dataset = os.path.join(args.IconDataset, "png")
    test_dataset = os.path.join(args.Task3Dataset, "images")
    annotations_dataset = os.path.join(args.Task3Dataset, "annotations")
    icon_images = os.listdir(icon_dataset)
    test_images = os.listdir(test_dataset)

Code (lines 27-32 of C-tasks/task_c3_utils.py):

    def extract_icon_features(icon_images, icon_dataset, pyramid_levels):
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.04, edgeThreshold=10)
    icon_features = {}

    for icon in icon_images:
        img = cv2.imread(os.path.join(icon_dataset, icon))

Code (lines 47-50 of C-tasks/task_c3_utils.py):

    def extract_test_features(test_image, test_dataset, upscale_factor):
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.04, edgeThreshold=10)

    img = cv2.imread(os.path.join(test_dataset, test_image))

# C-Task Execution
## 1. Setup Datasets
Follow instructions in [Additional Setup Instructions](#additional-setup-instructions) for loading C task datasets

## 2. Go to C-task folder
From the project root:
```
cd C-tasks
```

## 3. Execute Command for Respective Task
### Task C1 Example Command:
```
python main.py --Task1Dataset Task1Dataset
```

### Task C2 Example Command:
```
python main.py --IconDataset IconDataset --Task2Dataset Task2Dataset
```
Task C2 results are saved to C-tasks/c2_results/

### Task C3 Example Command:
```
python main.py --IconDataset IconDataset --Task3Dataset Task3Dataset
```
Task C3 results are saved to C-tasks/Task3Dataset/c3_results/

**Important note** - Task C3 uses the second version of the Task3Dataset i.e., the latest release which uses annotations in the format: [left,top,right,bottom]

C3 ground truths are plotted as AABBs - using their annotations [left,top,right,bottom] as coordinates, where:

(left,top) = top-left corner 

(right, bottom) = bottom-right corner


