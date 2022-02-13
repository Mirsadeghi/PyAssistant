
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