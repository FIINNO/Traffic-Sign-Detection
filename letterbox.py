import math
import cv2

def letterbox(img, boxes=None, new_shape=(640, 640), normalized=True):
    original_shape = img.shape[:2]
    ratio = min(new_shape[0] / original_shape[0], new_shape[1] / original_shape[1])
    new_unpad = (int(round(original_shape[1] * ratio)), int(round(original_shape[0] * ratio)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    resized_img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    dw /= 2
    dh /= 2
    top, bottom = math.floor(dh), math.ceil(dh)
    left, right = math.floor(dw), math.ceil(dw)

    img_padded = cv2.copyMakeBorder(resized_img, top, bottom, left, right,
                                    borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114))


    if boxes is not None:
        if normalized:
            boxes[:, 0] *= original_shape[1]
            boxes[:, 1] *= original_shape[0]
            boxes[:, 2] *= original_shape[1]
            boxes[:, 3] *= original_shape[0]

        boxes[:, 0] = boxes[:, 0] * ratio + dw  # x
        boxes[:, 1] = boxes[:, 1] * ratio + dh  # y
        boxes[:, 2] = boxes[:, 2] * ratio       # w
        boxes[:, 3] = boxes[:, 3] * ratio       # h

        if normalized:
            boxes[:, 0] /= new_shape[1]
            boxes[:, 1] /= new_shape[0]
            boxes[:, 2] /= new_shape[1]
            boxes[:, 3] /= new_shape[0]


    return img_padded, boxes, (top, left)
