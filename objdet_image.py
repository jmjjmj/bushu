# -*- coding: utf-8 -*-

import cv2
import torch
import numpy as np
import time
from onnxruntime import InferenceSession

from utils import load_classes, preprocess, postprocess, plot_results

if __name__ == "__main__":
    
    # get input data
    input_size = 640
    img_file = "./data/street.jpg"
    img = cv2.imread(img_file)
    img_h = img.shape[0]
    img_w = img.shape[1]
    input_data, offset = preprocess(img, input_size, input_size)

    # get classes
    class_name_file = "./data/coco.names"
    classes = load_classes(class_name_file)
    
    start_time = time.time()
    # inference
    output_names = ["output0"]
    onnx_file = "./yolov5s-fp16.onnx"
    session = InferenceSession(onnx_file)
    outputs = session.run(output_names=output_names, input_feed={"images": input_data})

    # postprocess
    pred_boxes, pred_cls, pred_scores = postprocess(outputs, 
                                                    img_w, 
                                                    img_h, 
                                                    input_size, 
                                                    input_size, 
                                                    offset, 
                                                    conf_thresh=0.25, 
                                                    iou_thresh=0.45)
    end_time = time.time()
    delta_time = end_time-start_time
    print(delta_time)
    # show result
    save_path = "./output/pred_result_onnx.jpg"
    plot_results(img, pred_boxes, pred_cls, pred_scores, class_names=classes, color=None, save_path=save_path)
