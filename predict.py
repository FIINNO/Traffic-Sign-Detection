import torch
from letterbox import letterbox
import cv2
from ultralytics.utils.ops import non_max_suppression
from evaluate import visualize_pred_vs_gt
from ultralytics.nn.tasks import DetectionModel



def predict(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_padded, _ = letterbox(img)
    img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
    assert img_tensor.shape == (3, 640, 640), f"Got {img_tensor.shape}"
    img_tensor = img_tensor.unsqueeze(0)
    print(img_tensor.shape)

    model.eval()
    with torch.no_grad():
        output = model.forward(img_tensor)
        nms_results = non_max_suppression(
            output[0],
            conf_thres=0.1,
            iou_thres=0.5,
            max_det=1000
        )
        print(nms_results)


    visualize_pred_vs_gt(img=img_tensor[0], nms=nms_results)




# checkpoint = torch.load('yolo11l.pt')
# cfg = checkpoint['model'].yaml

cfg = torch.load('yolol_cfg.yaml')
model = DetectionModel(cfg=cfg)
model.load_state_dict(torch.load('yolol_checkpoint_best.pth', map_location=lambda storage, loc: storage))

img = cv2.imread('./img_6.png')
predict(img, model)






