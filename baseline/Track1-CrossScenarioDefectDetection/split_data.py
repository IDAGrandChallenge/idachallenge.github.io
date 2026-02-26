"""
 1. 统计每类缺陷有多少个
 2. 划分数据集和验证集（每个类别挑选两百张图片进行训练），训练集和验证集按照8：2划分
 图片大小 512 * 512
"""

import os
import shutil
import yaml
import random
import json

if __name__ == '__main__':
    img_dir = '/home/shared/data/IDA比赛/分级赛道/NG_1156/'
    label_dir = '/home/shared/data/IDA比赛/分级赛道/NG_1156/'
    label_shuffix = ".txt"
    classes_list_dir = "/home/shared/data/IDA比赛/跨场景赛道/缺陷列表信息.txt"
    save_dir = '/home/lss/data1/'
    K = 20

    with open(classes_list_dir, 'r') as f:
        classes_list = [i.strip() for i in f.readlines()]


    classes_dict = {}
    for class_name in classes_list:
        classes_dict[class_name] = 0

    # 统计标签信息
    img_dict = {}
    for file_name in os.listdir(img_dir):
        if file_name.endswith('.bmp'):
            img_path = os.path.join(img_dir, file_name)
            label_path = os.path.join(img_dir, file_name.split('.')[0] + label_shuffix)
            if not os.path.join(label_path):
                continue
            with open(label_path, 'r') as f:
                if label_shuffix == '.txt':
                    lines = f.readlines()
                    for line in lines:
                        class_name = classes_list[int(line.strip().split(" ")[0])]
                        classes_dict[class_name] += 1
                        if class_name not in list(img_dict.keys()):
                            img_dict[class_name] = [file_name]
                        else:
                            img_dict[class_name].append(file_name)
                elif label_shuffix == '.json':
                    lines = json.load(f)
                    for line in lines:
                        class_name = line['class']
                        classes_dict[class_name] += 1
                        if class_name not in list(img_dict.keys()):
                            img_dict[class_name] = [file_name]
                        else:
                            img_dict[class_name].append(file_name)


    print("缺陷类别统计\n", classes_dict)
    classes_index_dict = dict(enumerate(classes_list))
    dataset_dict = {"path": save_dir,
                     "train": "images/train",
                     "val": "images/val", 
                     "test": "images/test",
                     "nc": len(classes_list),
                     "names":classes_index_dict}
    with open(os.path.join(save_dir, 'dataset.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(
            dataset_dict,
            f,
            allow_unicode=True,
            sort_keys=False  # 保持字段顺序
        )

    # 划分训练集，验证集，测试集
    # 从数据集中每个类别抽取20张图作为测试集（共140张），其余的数据作为训练集，训练集中train和val按8:2划分
    train_img_list = []
    val_img_list = []
    test_img_list = []
    for cls, img_list in img_dict.items():
        chosen_idx = random.sample(range(len(img_list)), 20) if len(img_list) > 20 else list(range(20))
        test_img_list.extend([img_list[i] for i in chosen_idx])
        unchosen_idx = list(set(list(range(len(img_list)))) - set(chosen_idx))
        train_img_list.extend([img_list[i] for i in unchosen_idx[:int(len(unchosen_idx)*0.8)]]) 
        val_img_list.extend([img_list[i] for i in unchosen_idx[int(len(unchosen_idx)*0.8)+1:]])

    # 数据集另存为
    save_label_dir = os.path.join(save_dir, "labels")
    save_image_dir = os.path.join(save_dir, "images")
    train_img_dir = os.path.join(save_dir, "images", "train")
    val_img_dir = os.path.join(save_dir, "images", "val")
    test_img_dir = os.path.join(save_dir, "images", "test")
    train_label_dir = os.path.join(save_dir, "labels", "train")
    val_label_dir = os.path.join(save_dir, "labels", "val")
    test_label_dir = os.path.join(save_dir, "labels", "test")
    if not os.path.exists(save_label_dir):
        os.makedirs(save_label_dir)
    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    for img_file in train_img_list:
        shutil.copy(os.path.join(img_dir, img_file), os.path.join(train_img_dir, img_file))
        label_file = img_file.split(".")[0] + label_shuffix
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(train_label_dir, label_file))
    for img_file in val_img_list:
        shutil.copy(os.path.join(img_dir, img_file), os.path.join(val_img_dir, img_file))
        label_file = img_file.split(".")[0] + label_shuffix
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(val_label_dir, label_file))
    for img_file in test_img_list:
        shutil.copy(os.path.join(img_dir, img_file), os.path.join(test_img_dir, img_file))
        label_file = img_file.split(".")[0] + label_shuffix
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(test_label_dir, label_file))
    print("split data finished!")