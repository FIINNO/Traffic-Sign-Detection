import os
import json
import shutil
import random


def extract_classes():
    annotation_folder = './annotations/'
    counter = 0

    class_dict = {}
    index = 0

    for filename in os.listdir(annotation_folder):
        with open(os.path.join(annotation_folder, filename), 'r') as file:
            data = json.load(file)

            for obj in data['objects']:
                label = obj['label']

                if label not in class_dict:
                    class_dict[label] = index
                    index += 1
        counter += 1
        if counter % 500 == 0:
            print(counter)

    with open('classes.json', 'w') as file:
        json.dump(class_dict, file, indent=2)


def sort_classes():
    file_path = './classes.json'
    with open(file_path, 'r') as file:
        class_dict = json.load(file)
        print(class_dict)

    grouped_dict = dict(sorted(class_dict.items()))

    with open('grouped_classes.json', 'w') as file:
        json.dump(grouped_dict, file, indent=2)


    sorted_dict = {}
    index = 0
    for key, value in grouped_dict.items():
        sorted_dict[key] = index
        index += 1

    with open('sorted_classes.json', 'w') as file:
        json.dump(sorted_dict, file, indent=2)


# Write class distribution to a .txt
def extract_class_distribution():
    with open('sorted_classes.json', 'r') as file:
        class_dict = json.load(file)
        class_distribution = {key: 0 for key in class_dict}

        annotation_folder = './annotations/'

        counter = 0
        for filename in os.listdir(annotation_folder):
            with open(os.path.join(annotation_folder, filename), 'r') as file:
                data = json.load(file)

                for obj in data['objects']:
                    label = obj['label']
                    class_distribution[label] += 1

                counter += 1
                if counter % 500 == 0:
                    print(counter)

        with open('class_distribution.json', 'w') as file:
            json.dump(class_distribution, file, indent=2)


# Convert annotations to yolo-format
def convert_to_yolo():
    annotation_folder = '../annotations/'
    output_folder = '../annotations_yolo/'

    with open('class_mapping.json', 'r') as file:
        class_dict = json.load(file)
    with open('grouped_class_mapping.json', 'r') as file:
        group_mapping = json.load(file)
        label_to_group = {
            label: group_name
            for group_name, labels in group_mapping.items()
            for label in labels
        }



    counter = 0
    for filename in os.listdir(annotation_folder):
        with open(os.path.join(annotation_folder, filename), 'r') as file:
            data = json.load(file)
            yolo_annotations = []

            width = data['width']
            height = data['height']


            for obj in data['objects']:
                label = obj['label']
                group_label = label_to_group[label]
                class_id = class_dict[group_label]
                w = (obj['bbox']['xmax'] - obj['bbox']['xmin'])
                h = (obj['bbox']['ymax'] - obj['bbox']['ymin'])
                x_center = (obj['bbox']['xmin'] + w/2) / width
                y_center = (obj['bbox']['ymin'] + h/2) / height
                w = w / width
                h = h / height
                yolo_annotations.append((class_id, x_center, y_center, w, h))

            output_path = output_folder + os.path.splitext(filename)[0] + '.txt'

            with open(output_path, 'w') as file:
                for annotation in yolo_annotations:
                    line = ' '.join(map(str, annotation))
                    file.write(line + '\n')
            counter += 1
            if counter % 500 == 0:
                print(counter)


def create_test_split():
    image_train_dir = "dataset_light/images/train"
    label_train_dir = "dataset_light/labels/train"
    image_test_dir = "dataset_light/images/test"
    label_test_dir = "dataset_light/labels/test"
    test_ratio = 0.2

    os.makedirs(image_test_dir, exist_ok=True)
    os.makedirs(label_test_dir, exist_ok=True)

    image_filenames = [f for f in os.listdir(image_train_dir)
                       if os.path.isfile(os.path.join(image_train_dir, f))]


    random.shuffle(image_filenames)
    test_size = int(len(image_filenames) * test_ratio)
    test_images = image_filenames[:test_size]

    for image_file in test_images:
        base_name, _ = os.path.splitext(image_file)

        image_src = os.path.join(image_train_dir, image_file)
        label_src = os.path.join(label_train_dir, base_name + ".txt")

        image_dst = os.path.join(image_test_dir, image_file)
        label_dst = os.path.join(label_test_dir, os.path.basename(label_src))

        if os.path.exists(label_src):
            shutil.move(image_src, image_dst)
            shutil.move(label_src, label_dst)
        else:
            print(f"Warning: Label for {image_file} not found. Skipping.")

    print(f"Moved {len(test_images)} image-label pairs to test split.")


create_test_split()

