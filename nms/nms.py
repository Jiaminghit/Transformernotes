import torch
# box = [x1, y1, x2, y2]
# boxes = [[x1, y1, x2, y2], 
#          [x1, y1, x2, y2],
#          [x1, y1, x2, y2]]

def iou(box, boxes, isMin = False):
    # 求bounding box 的面积
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # 求交集
    xx1 = torch.maximum(box[0], boxes[:, 0])
    yy1 = torch.maximum(box[1], boxes[:, 1])
    xx2 = torch.minimum(box[2], boxes[:, 2])
    yy2 = torch.minimum(box[3], boxes[:, 3])
    
    # 求交集的 area
    w = torch.maximum(torch.Tensor([0]), xx2 - xx1)
    h = torch.maximum(torch.Tensor([0]), yy2 - yy1)
    over_area = w * h
    
    # 
    if isMin:
        return over_area / torch.min(box_area, boxes_area)
    else :
        return over_area / (box_area + boxes_area - over_area)

def nms(boxes, thresh = 0.3, isMin = False):
    new_boxes = boxes[boxes[:, 0].argsort(descending=True)]
    keep_boxes = []
    while len(new_boxes) > 0:
        _box = new_boxes[0]
        keep_boxes.append(_box)
        if len(new_boxes) > 1 :
            _boxes = new_boxes[1:]
            new_boxes = _boxes[torch.where(iou(_box, _boxes, isMin) < thresh)]
        else :
            break
    return torch.stack(keep_boxes)

if __name__ == '__main__':
    box = torch.tensor([0, 0, 4, 4])
    boxes = torch.tensor([[0.5, 4, 4, 5, 5], [0.9, 1, 1, 5, 5], [0.4, 1, 1, 5, 5]])
    nms(boxes=boxes)