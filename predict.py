import torch
from letterbox import letterbox
import cv2
from ultralytics.utils.ops import non_max_suppression, xyxy2xywh
import time



def predict(img, model):

    img_tensor, padding = pre_process(img)
    model.eval()

    start_time = time.time()
    with torch.no_grad():
        output = model.forward(img_tensor)

    end_time = time.time()
    inference_time = end_time - start_time
    preds = post_process(output)

    img_np = img_tensor[0].permute(1, 2, 0).numpy()  # convert image to numpy array (H, W, C)
    return img_np, preds, padding, inference_time


def pre_process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_padded, _, padding = letterbox(img)
    img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
    assert img_tensor.shape == (3, 640, 640), f"Got {img_tensor.shape}"
    img_tensor = img_tensor.unsqueeze(0)
    #print(img_tensor.shape)
    return img_tensor, padding

def post_process(preds):
    nms_results = non_max_suppression(
            preds[0],
            conf_thres=0.1,
            iou_thres=0.5,
            max_det=1000
        )

    preds = []
    img_size = 640
    if len(nms_results[0]) > 0:
        for pred in nms_results[0]:
            bbox = xyxy2xywh(pred[:4].unsqueeze(0))[0]
            bbox[0] /= img_size
            bbox[1] /= img_size
            bbox[2] /= img_size
            bbox[3] /= img_size
            conf = pred[4].item()
            cls = int(pred[5].item())
            preds.append((bbox, conf, cls))

    return preds

def benchmark_inferece(image_list, model):
    total_time = 0
    for img_np in image_list:
        _, _, _, inference_time = predict(img_np, model)
        total_time += inference_time

    avg_time = total_time / len(image_list)
    fps = 1.0 / avg_time
    return avg_time, fps
