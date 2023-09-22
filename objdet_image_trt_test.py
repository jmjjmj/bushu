# -*- coding: utf-8 -*-

import cv2
import torch
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import time

from utils import load_classes, preprocess, postprocess, plot_results

if __name__ == "__main__":
    
    # get input data
    input_size = 640
    img_file = "/root/AI/deep_learning/yolov5/demo/tensorRT/data/dog.jpg"
    img = cv2.imread(img_file)
    img_h = img.shape[0]
    img_w = img.shape[1]
    start_time = time.time()
    input_data, offset = preprocess(img, input_size, input_size)
    end_time = time.time()
    preprocess_time = (end_time - start_time) * 1000
    print("preprocess time (ms): ", preprocess_time)
    input_dim = input_data.shape

    # get classes
    class_name_file = "/root/AI/deep_learning/yolov5/demo/tensorRT/data/coco.names"
    classes = load_classes(class_name_file)


    # deserialize model engine
    logger = trt.Logger(trt.Logger.WARNING)
    engine_path = "/root/AI/deep_learning/yolov5/demo/tensorRT/yolov5m-fp16.engine"
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    
    # inference
    start_time = time.time()
    with engine.create_execution_context() as context:
        context.set_binding_shape(engine.get_binding_index("input_data"), input_dim)
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in engine:
            binding_idx = engine.get_binding_index(binding)

            size = trt.volume(engine.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding_idx))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if engine.binding_is_input(binding):
                host_mem = np.ascontiguousarray(input_data)
                inputs.append([host_mem, device_mem])
            else:
                outputs.append([host_mem, device_mem])
        
        for i in range(len(inputs)):
            cuda.memcpy_htod_async(inputs[i][1], inputs[i][0], stream)

        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        for i in range(len(outputs)):
            cuda.memcpy_dtoh_async(outputs[i][0], outputs[i][1], stream)
        stream.synchronize()
        end_time = time.time()
        process_time = (end_time - start_time) * 1000
        print("process_time (ms): ", process_time)
      
        # postprocess
        start_time = time.time()
        output = outputs[i][0]
        output_dim = [25200, 85]
        output = output.reshape(output_dim)
        pred_boxes, pred_cls, pred_scores = postprocess(output, offset)
        end_time = time.time()
        postprocess_time = (end_time - start_time) * 1000
        print("postprocess_time (ms): ", postprocess_time)

        # show result
        save_path = "./pred_tensorRT.jpg"
        plot_results(img, pred_boxes, pred_cls, pred_scores, class_names=classes, color=None, save_path=save_path)
