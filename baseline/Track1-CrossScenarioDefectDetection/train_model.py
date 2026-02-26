from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"/home/ultralytics/ultralytics/cfg/models/v8/yolov8-seg.yaml")

    train_results = model.train(
        data=r"/home/data1/dataset.yaml",
        project=r"/home/data1/models/",
        epochs=2000,
        imgsz=512,
        batch=32,
        device="0",
    )