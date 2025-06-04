import streamlit as st
import torch
from ultralytics.nn.tasks import DetectionModel
from predict import predict, benchmark_inferece
from PIL import Image
import cv2
import numpy as np
import json
from ultralytics.utils.metrics import bbox_iou
from glob import glob
import requests
import os

model_url = "https://github.com/FIINNO/Traffic-Sign-Detection/releases/download/v1/checkpoint_V8_best.pth"
model_path = "yolo_checkpoint_best.pth"

if not os.path.exists(model_path):
    with open(model_path, 'wb') as f:
        f.write(requests.get(model_url).content)


@st.cache_resource
def load_class_map():
    with open("class_mapping.json", "r") as f:
        class_map = json.load(f)
    return {v: k for k, v in class_map.items()} # invert from key-value to value-key

@st.cache_resource
def load_model():
    cfg = torch.load("yolol_cfg.yaml")
    model = DetectionModel(cfg=cfg)
    model.load_state_dict(torch.load("yolo_checkpoint_best.pth", map_location=lambda storage, loc: storage))
    return model

def draw_boxes(img, preds):
    h, w, _ = img.shape
    detected_signs = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for pred_box, pred_conf, pred_cls in preds:
        x_center, y_center, width, height = pred_box.tolist()
        x_min = int((x_center - width / 2) * w)
        y_min = int((y_center - height / 2) * h)
        x_max = int((x_center + width / 2) * w)
        y_max = int((y_center + height / 2) * h)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

        crop = img[y_min:y_max, x_min:x_max].copy()

        class_name = index_to_class.get(int(pred_cls), f"Class {int(pred_cls)}")
        detected_signs.append({
            "crop": crop,
            "cls": class_name,
            "conf": pred_conf
        })

    return img, detected_signs

def filter_preds(preds, iou_threshold = 0.5):
    if not preds:
        return []

    boxes = torch.stack([p[0] for p in preds])
    conf = torch.tensor([p[1] for p in preds])
    cls = torch.tensor([p[2] for p in preds])

    keep = []
    suppressed = set()
    for i in range(len(preds)):
        if i in suppressed:
            continue
        keep.append(i)
        for j in range(i + 1, len(preds)):
            if j in suppressed:
                continue
            iou = bbox_iou(boxes[i].unsqueeze(0), boxes[j].unsqueeze(0), xywh=True)
            if iou > iou_threshold:
                if conf[i] >= conf[j]:
                    suppressed.add(j)
                else:
                    suppressed.add(i)

    return [preds[i] for i in keep]

model = load_model()

index_to_class = load_class_map()

st.title("Traffic Sign Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image, preds, padding, inference_time = predict(image, model)
    preds = filter_preds(preds)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_image, detected_signs = draw_boxes(image, preds)
    height, width, _ = result_image.shape
    result_image = result_image[padding[0]:height - padding[0], padding[1]:width - padding[1], :]
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Image")
        st.image(result_image, use_container_width=True, clamp=True)
        st.markdown(f"##### Inference Time: {inference_time:.3f}s")
    with col2:
        st.markdown("### Predictions")
        for sign_info in detected_signs:
            st.image(sign_info["crop"], caption=(f"{sign_info['cls']} ({sign_info['conf']:.2f})"), use_container_width=True, clamp=True)
