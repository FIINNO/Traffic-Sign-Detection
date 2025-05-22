from ultralytics.utils.ops    import non_max_suppression, xyxy2xywh, xywh2xyxy
from ultralytics.utils.metrics import bbox_iou
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches





def process_batches(batch_list):
    """
    Extracts data instances from each batch into a list.
    Args:
        batch_list (List[batch]): List of batches containing ground truths

    Returns:
        (List[dict]): List of dicts containing object ground truths for each dataset instance.
    """

    batch_size = batch_list[0]['img'].shape[0]
    labels = []
    for i in range(len(batch_list)):
        for img_index in range(batch_size):
            mask = batch_list[i]['batch_idx'] == img_index
            image_labels = {
               'bboxes': batch_list[i]['bboxes'][mask],
                'cls': batch_list[i]['cls'][mask],
            }
            labels.append(image_labels)

    return labels


def compute_metrics(preds, batch_list, iou_threshold=0.5, img_size=640):
    """
    Computes recall and precision of model predictions
    Args:
        preds (List[tensor]): List of model predictions
        batch_list (List[batch]): List of batches containing ground truths
        iou_threshold (float): IoU threshold
        img_size (int): Image size

    Returns:

    """
    labels = process_batches(batch_list)

    tp = 0
    fp = 0
    total_objects = 0

    # Loop through prediction and gt for each data instance
    for pred, gt in zip(preds, labels):

        nms_results = non_max_suppression(  # nms
            pred,
            conf_thres=0.25,
            iou_thres=0.5,
            max_det=1000
        )

        pred_boxes = nms_results[0][:,:4]
        pred_conf = nms_results[0][:,4]
        pred_classes = nms_results[0][:,5]

        # Convert bboxes from x1y1x2y2 to xywh and normalize
        pred_boxes_xywh = xyxy2xywh(pred_boxes)
        pred_boxes_xywh[:, 0] /= img_size
        pred_boxes_xywh[:, 1] /= img_size
        pred_boxes_xywh[:, 2] /= img_size
        pred_boxes_xywh[:, 3] /= img_size


        gt_boxes = gt['bboxes']
        gt_classes = gt['cls']
        total_objects += len(gt_boxes)

        matched = []
        for pred_box, pred_class in zip(pred_boxes_xywh, pred_classes):

            match_found = False
            for i in range(len(gt_boxes)):
                if i in matched:
                    continue
                if pred_class != gt_classes[i]:
                    continue
                iou = bbox_iou(pred_box, gt_boxes[i])
                if iou >= iou_threshold:
                    matched.append(i)
                    match_found = True
                    break
            if match_found:
                tp += 1
            else:
                fp += 1


    fn = total_objects - tp
    print(f"Total tp: {tp}, fp: {fp}, fn: {fn}")
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (total_objects + 1e-9)
    print(f"Precision: {precision}, Recall: {recall}")





def evaluate_yolo(model, test_loader):
    # evaluate on test dataset
    total_gt = []
    total_preds = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            total_gt.append(batch)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            images = batch['img']
            output = model.forward(images)

            for i in range(output[0].size(0)):
                total_preds.append(output[0][i].unsqueeze(0).cpu())

    compute_metrics(total_preds, total_gt)



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
