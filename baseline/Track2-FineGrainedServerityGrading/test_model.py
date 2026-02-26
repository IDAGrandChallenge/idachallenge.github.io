from ultralytics import YOLO

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