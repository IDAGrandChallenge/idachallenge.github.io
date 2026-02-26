## 1. Overview of the Approach
- Beyond basic detection and segmentation accuracy, real-world industrial defects are often graded based on morphological 
traits like area, dimensions, and contrast. Our approach maps these features to standardized severity tiers: Acceptable,
Marginal NG, NG, and Gross NG. The model's effectiveness is then evaluated by the alignment between its predicted grades
and these industry standards.
- The baseline pipeline starts with a YOLOv8-seg model for instance segmentation. It then calculates a severity score 
by applying a weighted combination of four metrics extracted from the masks: defect area, bounding box length/width,
and mean grayscale contrast. Ultimately, this score is used to classify the final defect grade.
## 2. Environment Setup
- Runtime requirements: Python3.X, Pytorch2.x, GPU
- Below is an example of setting up a training environment with Python 3.12, Torch 2.10, and GPU:
1. Create a new Anaconda virtual environment
> conda create -n train python=3.12
2. Activate the environment
> conda activate train
3. Install dependencies
> pip install -r requirements.txt

## 3. Data Preparation
- The dataset contains 1154 images and seven defect types: Stain, Scratch, Bubble, Particle, Dent, Damage, and Chipping.
- Original data files
    ```` text
  Severity Grading Track/
        ├── images
        ├── grading_labels
        └── defect_list.txt     
- Data format:
The annotations for each image is stored in a single .json file. Each defect is represented as a dictionary:
`{"class":"class_id", "points":"x1, y1, x2, y2 ... xn, yn", "severity":"grade"}`
  - Where:
    - class is the defect category,
    - points is the contour point set; coordinates are normalized polygon vertex coordinates (relative to image width and height),
    - severity is the defect grade (Acceptable, Marginal NG, NG, or Gross NG).

- Train/test split: For each class, 20 images are sampled as the test set; the remaining data are used for training. 
The training set is further split into training and validation subsets at an 8:2 ratio.
- Run the script `split_data.py` to split the data.
> python split_data.py

- Data after splitting 
  ```` text
  dataset/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/      
        ├── labels/
        │   ├── train/
        │   ├── val/
        │   └── test/      
        └── data.yaml      # data configuration file
- images/train, images/val, images/test：store images
- labels/train, labels/val, labels/test：store corresponding label files (same-name .txt files)

## 4. Training
- Configure the training parameters, then start training.
```python
# train_model.py
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"/home/ultralytics/ultralytics/cfg/models/v8/yolov8-seg.yaml")
    train_results = model.train(
        data=r"/home/data/dataset.yaml",                                         
        project=r"/home/data/models/",                                        
        epochs=2000,      
        imgsz=512,            
        batch=32,              
        device="0",  
    )             

```

> python train_model.py
## 5. Validation and Testing
- Evaluation metric: STrack2 = 0.2 × Sloc + 0.2 × Scls + 0.6 × Sgrade
  - Localization (Sloc): Measured by Mean Intersection over Union (mIoU).
    - Sloc = mean(IoU(Pi, Gi))
  - Classification (Scls): Measured by Macro-F1 Score for defect type identification.                  
    - Scls = 0.5 × Sf ine + 0.5 × Sscreen
  - Screening Efficiency (Sscreen): A composite of Image-Level Recall and Specificity to penalize false
  alarms in unseen domains.
    - Sscreen = 0.5 × Recallimg + 0.5 × Specificityimg
  - Severity Grading (Sgrade): The core metric, measured by Quadratic Weighted Kappa (QWK) to
  penalize misalignment between predicted and actual severity levels.

- Model inference: Use the trained model to run inference on the test set and generate predicted labels.
> python test_model.py
```python
# test_model.py
from ultralytics import YOLO
import os

model = YOLO("/home/data/models/train/weights/best.pt")

image_dir = "/home/data/images/test/"
data_dir = "/home/data/dataset.yaml"
save_dir = "/home/data/result/"
os.makedirs(save_dir, exist_ok=True)

results = model.predict(
    source=image_dir,
    save=True,            
    save_txt=True,       
    save_conf=False,       
    project=save_dir,
    name="",              
    exist_ok=True,
    save_dir=True,     
    vid_stride=1
)
```
- Generate severity grading labels based on the model predicted results.
> python classify_grades.py

- Metric calculation
    - Using the metric formulas, run the script to compute the evaluation metrics.
> python caculate_metric.py
```python
if __name__ == '__main__':
    gt_img_dir = "/home/data/images/test/"
    gt_txt_dir = "/home/data/test_grading/defects grading/"
    pt_img_dir = "/home/data/result/predict/"
    pt_txt_dir = "/home/data/result/predict_grading/defects grading/"
    cls_txt_dir = "/home/data/IDA/Grading/defect_list.txt"
    txt_shuffix = '.json'
    cm = CaculateMetric()
    SDict = cm.process_data(gt_img_dir, gt_txt_dir, pt_img_dir, pt_txt_dir, cls_txt_dir, txt_shuffix, S=2)
    print("caculate finished!")
   ```

- Test results

|          | image-level  | image-level | image-level | instance-level | instance-level | instance-level | total   |
|----------|--------------|-------------|-------------|----------------|----------------|----------------|---------|
| class    | Sloc         | Scls        | Sgrade      | Sloc           | Scls           | Sgrade         | Strack2 |
| all      | /            | 0.232       | /           | 0.590          | 0.681          | 0.311          | 0.441   |
| Stain    | /            | 0.179       | /           | 0.648          | 0.491          | 0.462          | 0.505   |
| Scratch  | /            | 0.234       | /           | 0.557          | 0.678          | 0.494          | 0.543   |
| Bubble   | /            | 0.250       | /           | 0.608          | 0.750          | 0.222          | 0.405   |
| Particle | /            | 0.250       | /           | 0.618          | 0.738          | 0.353          | 0.483   |
| Dent     | /            | 0.224       | /           | 0.755          | 0.660          | 0.030          | 0.301   |
| Damage   | /            | 0.236       | /           | 0.482          | 0.722          | 0.333          | 0.441   |
| Chipping | /            | 0.250       | /           | 0.709          | 0.726          | 0.277          | 0.453   |
