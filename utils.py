# -*- coding: utf-8 -*-

import cv2
import torch
import numpy as np

# colors for visualization
COLORS = [[0, 0, 255], [80, 127, 255], [0, 255, 0],
          [255, 255, 0], [48, 64, 33], [205, 224, 64],
          [255, 0, 0], [0, 255, 255], [226, 43, 138]]

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def preprocess(img, height, width):
    h_ratio = height / img.shape[0]
    w_ratio = width/ img.shape[1]
    ratio = min(w_ratio, h_ratio)
    border_h = int(img.shape[0] * ratio)
    border_w = int(img.shape[1] * ratio)
    x_offset = int((width - border_w) / 2)
    y_offset = int((height - border_h) / 2)
    offset = [x_offset, y_offset, ratio]

    input_data = cv2.resize(img, (border_w, border_h))
    input_data = cv2.copyMakeBorder(input_data, y_offset, y_offset, x_offset, x_offset, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    input_data = np.array(input_data, dtype=np.float32)
    input_data = input_data / 255.0
    input_data = input_data.transpose((2, 0, 1))
    input_data = input_data[np.newaxis, :, :, :]

    return input_data, offset

def box_cxcywh_to_xyxy(cbox, offset):
    x_c = cbox[0]
    y_c = cbox[1]
    w = cbox[2]
    h = cbox[3]
    x_offset = offset[0]
    y_offset = offset[1]
    ratio = offset[2]
    scale = 1.0 / ratio
    xmin = (x_c - 0.5 * w - x_offset) * scale
    ymin = (y_c - 0.5 * h - y_offset) * scale
    xmax = (x_c + 0.5 * w - x_offset) * scale
    ymax = (y_c + 0.5 * h - y_offset) * scale
    
    box_coord = np.array([xmin, ymin, xmax, ymax])
    return box_coord

def clip_box(pred_box, img_w, img_h):
    pred_box[0] = np.clip(pred_box[0], 0, img_w)
    pred_box[1] = np.clip(pred_box[1], 0, img_h)
    pred_box[2] = np.clip(pred_box[2], 0, img_w)
    pred_box[3] = np.clip(pred_box[3], 0, img_h)

def rescale_bbox(pred_box, img_w, img_h, input_w, input_h):
    w_ratio = input_w / img_w
    h_ratio = input_h / img_h
    ratio = min(w_ratio, h_ratio)
    new_w = np.round(input_w * ratio)
    new_h = np.round(input_h * ratio)
    offset_w = (input_w - new_w) / 2
    offset_h = (input_h - new_h) / 2
    scale = 1.0 / ratio

    scale_box = np.array([0, 0, 0, 0])
    scale_box[0] = (pred_box[0] - offset_w) * scale
    scale_box[1] = (pred_box[1] - offset_h) * scale
    scale_box[2] = (pred_box[2] - offset_w) * scale
    scale_box[3] = (pred_box[3] - offset_h) * scale

    clip_box(scale_box, img_w, img_h)
    return scale_box

def box_iou(box1, box2):
    xmin1 = box1[0]
    ymin1 = box1[1]
    xmax1 = box1[2]
    ymax1 = box1[3]

    xmin2 = box2[0]
    ymin2 = box2[1]
    xmax2 = box2[2]
    ymax2 = box2[3]

    # calculate intersection area
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    w = np.max([0.0, xx2 - xx1 + 1])
    h = np.max([0.0, yy2 - yy1 + 1])
    intersection_area = w * h

    # calculate union area
    area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
    union_area = area1 + area2 - intersection_area

    # calculate iou
    eps = 1e-8
    iou = intersection_area / (union_area + eps)
    return iou


def non_max_suppression(prediction, 
                        pred_boxes, 
                        pred_cls, 
                        pred_scores, 
                        offset, 
                        conf_thres=0.25, 
                        iou_thres=0.45,  
                        max_obj_num=100):
    
    det_boxes = []
    det_cls_idx = []
    det_cls_scores = []
    det_conf = []
    
    for i in range(len(prediction)):
        pred_item = prediction[i]
        obj_conf = pred_item[4]
        if obj_conf < conf_thres:
            continue

        cls_idx = np.argmax(pred_item[5:])
        cls_conf = pred_item[5 + cls_idx]
        conf = obj_conf * cls_conf
        if conf > conf_thres:
            cbox = pred_item[:4] 
            box = box_cxcywh_to_xyxy(cbox, offset)
            det_boxes.append(box)
            det_cls_idx.append(cls_idx)
            det_cls_scores.append(cls_conf)
            det_conf.append(conf)

    det_boxes = np.array(det_boxes)
    det_conf = np.array(det_conf)
    order = np.argsort(det_conf)[::-1]

    keep = []
    while len(order) > 0:

        if len(keep) > max_obj_num:
            break

        top1_idx = order[0]
        top1_box = det_boxes[top1_idx]
        keep.append(top1_idx)
        order = np.delete(order, 0)

        idx = 0
        inds = []
        for k in range(len(order)):
            topk_idx = order[k]
            topk_box = det_boxes[topk_idx]
            iou = box_iou(top1_box, topk_box)
            if iou <= iou_thres:
                inds.append(k)
        order = order[inds]

    for i in range(len(keep)):
        idx = keep[i]
        pred_boxes.append(det_boxes[idx])
        pred_cls.append(det_cls_idx[idx])
        pred_scores.append(det_cls_scores[idx])

    return
    
def postprocess(prediction, 
                img_w, 
                img_h, 
                input_w, 
                input_h, 
                offset, 
                conf_thresh=0.25, 
                iou_thresh=0.45, 
                max_det=300):
    
    prediction = prediction[0][0]
    pred_boxes = []
    pred_cls = []
    pred_scores = []
    rescale_boxes = []
    max_obj_num = max_det
    non_max_suppression(prediction, 
                        pred_boxes, 
                        pred_cls, 
                        pred_scores, 
                        offset, 
                        conf_thres=conf_thresh, 
                        iou_thres=iou_thresh,  
                        max_obj_num=max_obj_num)
    # for i in range(len(pred_boxes)):
    #     box = rescale_bbox(pred_boxes[i], img_w, img_h, input_w, input_h)
    #     rescale_boxes.append(box)

    return pred_boxes, pred_cls, pred_scores

def plot_results(img, boxes, classes, scores, class_names=None, color=None, save_path=''):
    img = np.copy(img)
    for i in range(len(boxes)):
        box = boxes[i]
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])

        score = scores[i]
        class_id = classes[i]
        text = f'{class_names[class_id]}: {score:0.2f}'
        color_id = class_id % len(COLORS)
        color_array = COLORS[color_id]
        img = cv2.putText(img, text, (xmin, ymin - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_array, 2)
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_array, 2)
        cv2.imwrite(save_path, img)

    return