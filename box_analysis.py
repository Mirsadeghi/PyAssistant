import torch, math
import numpy as np
from .utils import isClass
from .evaluation import hungarian_match
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # https://github.com/ultralytics/yolov5/blob/develop/utils/general.py
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def find_pair_boxes(BB, BBGT, ovthresh=.25):
    confidence = BB[:, 4]
    BB = BB[:, 0:4]
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]

    n = BBGT.shape[0]
    m = BB.shape[0]
    isUsed = np.zeros(n).astype('bool')
    # tp = np.zeros(m)
    # fp = np.zeros(m)
    gt_pair = np.zeros(m)-1
    for d, bb in enumerate(BB):
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
        ih = np.maximum(iymax - iymin + 1.0, 0.0)
        inters = iw * ih

        # union
        uni = (
            (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
            + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
            - inters
        )

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not isUsed[jmax]:
                # tp[d] = 1.0
                gt_pair[d] = jmax
                isUsed[jmax] = 1
        #     else:
        #         fp[d] = 1.0
        # else:
        #     fp[d] = 1.0
    return gt_pair
    '''
    ious = []
    for pred in preds:
        iou = []
        for gt_bbx in gt_bbxs:
            if isClass(pred, list):
                pred = np.array(pred).astype('float')
            pred = pred.astype('int')
            if isClass(gt_bbx, list):
                gt_bbx = np.array(gt_bbx)
            gt_bbx = gt_bbx.astype('int')
            iou.append(bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbx)))
            
        ious.append(np.array(iou))
    ious = np.array(ious)
    if ious.shape[0] > ious.shape[1]:
        idx_gt = ious.argmax(0)
        idx_pred = np.arange()
    else:
        macthed = hungarian_match(ious)
        idx_gt = [p[0] for p in macthed]
        idx_pred = [p[1] for p in macthed]
    return idx_pred, idx_gt, ious
    '''
# convert list of boxes to the polygon points
def box2poly(boxes, format='xyxy'):
    polys = []
    for box in boxes:
        if format == 'xywh':
            poly = [[box[0], box[1]], [box[0]+box[2], box[1]], [box[0]+box[2], box[1]+box[3]], [box[0], box[1]+box[3]], [box[0], box[1]]]
        elif format == 'xyxy':
            poly = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]], [box[0], box[1]]]
        else:
            raise('Undefined format')
        polys.append(poly)
    return polys