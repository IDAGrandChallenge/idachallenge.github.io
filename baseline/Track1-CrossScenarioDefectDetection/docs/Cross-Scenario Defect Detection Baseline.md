## 1. Overview of the Approach
- The product can be divided into multiple regions, and the defect types in each region are roughly similar. 
The goal is to train a vision model on a fixed region and still generalize well on other unseen regions.
- Specifically, the baseline trains an instance segmentation model and evaluates its segmentation performance 
in other cross-domain regions. The segmentation model used here is YOLOv8-seg.

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
    Cross-Scenario Track/
        ├── images
        ├── seg_labels
        └── defect_list.txt      
- Data format: Labels are annotated in the YOLO segmentation format. Each image has a corresponding  .txt
label file. In each .txt file, each line represents one instance: `class_id x1 y1 x2 y2 x3 y3 ... xn yn`
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
    model = YOLO(r"/home/lss/ultralytics/ultralytics/cfg/models/v8/yolov8-seg.yaml")
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
- Evaluation metric: STrack1 = 0.3 × Sloc + 0.3 × Scls + 0.4 × Sscreen
  - Localization (Sloc): Measured by Mean Intersection over Union (mIoU).
    - Sloc = mean(IoU(Pi, Gi))
  - Classification (Scls): Measured by Macro-F1 Score for defect type identification.                  
    - Scls = 0.5 × Sf ine + 0.5 × Sscreen
  - Screening Efficiency (Sscreen): A composite of Image-Level Recall and Specificity to penalize false
  alarms in unseen domains.
    - Sscreen = 0.5 × Recallimg + 0.5 × Specificityimg

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

- Metric calculation
    - Using the metric formulas, run the script to compute the evaluation metrics.
> python caculate_metric.py
```python
if __name__ == '__main__':
    gt_img_dir = "/home/data/images/test/"
    gt_txt_dir = "/home/data/labels/test/"
    pt_img_dir = "/home/data/result/predict/"
    pt_txt_dir = "/home/data/result/predict/labels/"
    cls_txt_dir = "/home/data/IDA/Cross-Scenario/defect_list.txt"
    txt_shuffix = '.txt'
    cm = CaculateMetric()
    SDict = cm.process_data(gt_img_dir, gt_txt_dir, pt_img_dir, pt_txt_dir, cls_txt_dir, txt_shuffix, S=1)
    print("caculate finished!")
```
  
- Test results
  
|          | image-level | image-level |  image-level| instance-level | instance-level | instance-level | total   |
|----------|-------------|-------------|-----------|----------------|----------------|----------------|---------|
| class    | Sloc        | Scls        | Sscreen   | Sloc           | Scls           | Screen         | Strack1 |
| all      | /           | 0.231       | 0.461     | 0.646          | 0.448          | /              | 0.582   |
| Stain    | /           | 0.203       | 0.406     | 0.624          | 0.371          | /              | 0.522   |
| Scratch  | /           | 0.250       | 0.500     | 0.628          | 0.481          | /              | 0.608   |
| Bubble   | /           | 0.250       | 0.500     | 0.664          | 0.500          | /              | 0.624   |
| Particle | /           | 0.250       | 0.500     | 0.645          | 0.495          | /              | 0.617   |
| Dent     | /           | 0.191       | 0.382     | 0.814          | 0.371          | /              | 0.566   |
| Damage   | /           | 0.250       | 0.500     | 0.575          | 0.464          | /              | 0.587   |
| Chipping | /           | 0.217       | 0.435     | 0.727          | 0.455          | /              | 0.593   |



