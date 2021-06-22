import cv2
import numpy as np
import torch
import torchvision as tvs


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def mns(prediction, conf_thres=0.5, iou_thres=0.6):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    xc = prediction[..., 4] > conf_thres  # candidates

    output = [None] * prediction.shape[0]
    for i, x in enumerate(prediction):  # image index, image inference
        x = x[xc[i]]  # confidence
        if not x.shape[0]:
            continue
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[..., :4])

        conf = x[..., 4].view(-1, 1)
        x = torch.cat((box, conf), 1)

        # Batched NMS
        # boxes (offset by class), scores
        boxes, scores = x[..., :4], x[..., 4]
        ind = tvs.ops.nms(boxes, scores, iou_thres)
        output[i] = x[ind]

    return output


def preprocess(image: np.array, net_inshape: int = 640):
    h, w = image.shape[:2]
    d = max(h, w)
    dy = d - h
    dx = d - w
    img = cv2.copyMakeBorder(
        image, 0, dy, 0, dx, borderType=cv2.BORDER_CONSTANT, value=(113, 113, 113)
    )

    resize = float(net_inshape) / float(img.shape[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(
        img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR
    )
    img = np.float32(img)
    img = img / 255.0
    img = img.transpose(2, 0, 1)

    return img, resize
