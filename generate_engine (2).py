# -*- coding: utf-8 -*-

import tensorrt as trt

if __name__ == "__main__":

    # set batch size to 1
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    # create logger instance
    logger = trt.Logger()
    trt.init_libnvinfer_plugins(logger,"")

    # generate engine
    builder = trt.Builder(logger)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, logger)
    runtime = trt.Runtime(logger)
    config = builder.create_builder_config()

    config.max_workspace_size = 1 << 32  # 4GB
    builder.max_batch_size = 1

    onnx_file = "/root/AI/deep_learning/yolov5/demo/tensorRT/yolov5m.onnx"
    success = parser.parse_from_file(onnx_file)

    serialize_engine = builder.build_serialized_network(network, config)

    engine_path = "/root/AI/deep_learning/yolov5/demo/tensorRT/yolov5m.engine"
    with open(engine_path, "wb") as f:
        f.write(serialize_engine)
        print("generate engine successfully")

