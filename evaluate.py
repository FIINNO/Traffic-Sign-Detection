from ultralytics.utils.ops    import non_max_suppression, xyxy2xywh, xywh2xyxy
from ultralytics.utils.metrics import bbox_iou
import torch

def evaluate():
    model.eval()
    all_preds = []
    all_gts = []
    validation_loop = tqdm(val_loader, desc=f"Training epoch: {epoch+1}/{epochs}")
    for batch in validation_loop:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        images = batch["img"]
        gt_boxes = batch["bboxes"]
        gt_classes = batch["cls"]

        with torch.no_grad():
            output = model(images)

        nms_results = non_max_suppression(
            output,
            conf_thres=0.25,
            iou_thres=0.45,
        )
        print("NMS_RESULTS: ", nms_results)

        for i, pred in enumerate(nms_results):
            pred_boxes = pred[:, :4]
            pred_conf = pred[:, 4]
            pred_classes = pred[:, 5]

            pred_boxes_xywh = xyxy2xywh(pred_boxes)
            img_size = 640
            pred_boxes_xywh[:, 0] /= img_size
            pred_boxes_xywh[:, 1] /= img_size
            pred_boxes_xywh[:, 2] /= img_size
            pred_boxes_xywh[:, 3] /= img_size

            true_boxes = gt_boxes[i]
            true_classes = gt_classes[i]

            iou = bbox_iou(pred_boxes_xywh, true_boxes)
            #print("IOU", iou)

            all_preds.append((pred_boxes_xywh, pred_conf, pred_classes))
            all_gts.append((true_boxes, true_classes))
    #print(all_preds)
    #print(all_gts)
    return images, all_gts, all_preds

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_pred_vs_gt(img, gt, pred):
    h, w = img.shape[:2]
    gt_box = gt[0]
    gt_cls = gt[1]
    pred_box = pred[0]
    pred_conf = pred[1]
    pred_cls = pred[2]
    print("box:", pred_box)
    print("class:", pred_cls)
    print("conf:", pred_conf)

    img_np = img.permute(1, 2, 0).numpy()
    print("IMAGE", img_np)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_np)

    x_center, y_center, width, height = gt_box.tolist()

    # convert xywh to top-left corner (x_min, y_min)
    x_min = (x_center - width / 2) * img_np.shape[1]
    y_min = (y_center - height / 2) * img_np.shape[0]
    box_w = width * img_np.shape[1]
    box_h = height * img_np.shape[0]

    # create a rectangle patch
    rect = patches.Rectangle((x_min, y_min), box_w, box_h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    # draw each bounding box
    x_center, y_center, width, height = pred_box[0].tolist()

    # convert xywh to top-left corner (x_min, y_min)
    x_min = (x_center - width / 2) * img_np.shape[1]
    y_min = (y_center - height / 2) * img_np.shape[0]
    box_w = width * img_np.shape[1]
    box_h = height * img_np.shape[0]

    # create a rectangle patch
    rect = patches.Rectangle((x_min, y_min), box_w, box_h, linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)

    plt.title("Image with Resized Bounding Boxes")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
