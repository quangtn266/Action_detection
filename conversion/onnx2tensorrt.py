import tensorrt as trt
import numpy as np
#import pycuda.autoinit
#import pycuda.driver as cuda
import time

input_size = 256
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)

    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
     
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    builder.max_workspace_size = 1<<22 #1<<20
    # we have only one image in batch
    builder.max_batch_size = 1

    config = builder.create_builder_config()

    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True
    builder.int8_mode = True

    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    profile = builder.create_optimization_profile()
    dynamic_inputs = False
    for input in inputs:
        #log.info("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        if input.shape[0] == -1:
            dynamic_inputs = True
            dynamic_batch_size = "1,8,16"
            if dynamic_batch_size:
                if type(dynamic_batch_size) is str:
                    dynamic_batch_size = [int(v) for v in dynamic_batch_size.split(",")]
                    assert len(dynamic_batch_size) == 3
                    min_shape = [dynamic_batch_size[0]] + list(input.shape[1:])
                    opt_shape = [dynamic_batch_size[1]] + list(input.shape[1:])
                    max_shape = [dynamic_batch_size[2]] + list(input.shape[1:])
                    profile.set_shape(input.name, min_shape, opt_shape, max_shape)
                    #print(min_shape, opt_shape, max_shape, "#"*90)
                    #log.info("Input '{}' Optimization Profile with shape MIN {} / OPT {} / MAX {}".format(input.name, min_shape, opt_shape, max_shape))
            else:
                shape = [batch_size] + list(input.shape[1:])
                profile.set_shape(input.name, shape, shape, shape)
                #log.info("Input '{}' Optimization Profile with shape {}".format(input.name, shape))

    config.add_optimization_profile(profile)
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    engine = builder.build_engine(network,  config)
    print("Engine Created :", type(engine))
    
    return engine

if __name__ == "__main__":
    onnx_file_path = "./c2d_r50.onnx"
    engine = build_engine(onnx_file_path)
    context = engine.create_execution_context()
    print("Completed creating Engine")

    serialized_engine = engine.serialize()
    engine_path = "./c2d_r50.trt"
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

