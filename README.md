# Action_detection
The demo code for inference (YOWO)

This code is just onnx and trt conversion and inference for YOWO Pytorch.

Thanks for his contribution: https://github.com/yjh0410/PyTorch_YOWO

## Steps for onnx
1. onnx conversion: python torch2onnx.py [Arg]
2. To run inferrence with onnx model: python onnx_inference.py [Arg]

## Steps for trt.
To convert and run inference .trt code, you should convert onnx model and run it directly into Edge device. It means that both files for trt steps should be copied into the device.

1. trt conversion: python trt2onnx.py [Arg]
2. trt inference: python trt_inference.py [Arg]

Please, take a look the code before running, because it needs some argument.