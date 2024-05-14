import argparse
import tensorrt as trt
import numpy as np
import os

import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import time

from PackPathway import PackPathway

import torch
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)

from utils import softmax, softmax_stable, visualization_state

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 8
sampling_rate = 2
frames_per_second = 30
alpha = 4

transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            #PackPathway()
        ]
    ),
)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    
    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
        
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

                
                
    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
       
            
    def __call__(self,x:np.ndarray,batch_size=2):
        
        x = x.astype(self.dtype)
        
        np.copyto(self.inputs[0].host,x.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        
        self.stream.synchronize()
        return [out.host.reshape(batch_size,-1) for out in self.outputs]


def parse_args():
    parser = argparse.ArgumentParser(description='Loading .trt file of a pytorchvideo for a inference')
    parser.add_argument('--trt-file', type=str, help='.trt file', default=None)
    parser.add_argument('--sample-file', type=str, help='web camera or a video', default=None)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output', type=str, help='output file name', default=None)
    args = parser.parse_args()
    return args        
        


if __name__ == "__main__":
 
    args = parse_args()

    batchsize = args.batchsize
    trt_engine_path = args.trt_file

    model = TrtModel(trt_engine_path)
    shape = model.engine.get_binding_shape(0)
    retraining = True

    if args.sample_file == None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.sample_file)
        import json
        with open("kinetics_classnames.json", "r") as f:
            kinetics_classnames = json.load(f)
        
        kinetics_id_to_classname = {}
        for k, v in kinetics_classnames.items():
            kinetics_id_to_classname[v] = str(k).replace('"', "")

    clip=[]
    avg_fps = []
    pred_class_names = " "
    #packpathway = PackPathway()

    videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(args.output, fourcc, 10.0, (videoWidth, videoHeight), True)

    while retraining:
        retraining, frame =  cap.read()
        if not retraining and frame is None:
            continue

        clip.append(frame)
        if len(clip) == 5:
            
            # Getting input data with 32 frames (BxCxNxHxW).
            # B: Batchsize.
            # C: Channel.
            # N: Num of frames.
            # H: Height.
            # W: Width.

            inputs = np.array(clip).astype(np.float32)
            inputs = np.transpose(inputs, (3, 0, 1, 2))
            inputs = torch.from_numpy(inputs)

            video_data = {'video': inputs}
            video_data = transform(video_data)
            inputs = video_data["video"]
            
            #dummy_input = [i.to("cpu")[None, ...] for i in inputs]
            dummy_input = torch.unsqueeze(inputs, 0)
            start = time.time()
            #preds = model(dummy_input[0].cpu().detach().numpy(), dummy_input[1].cpu().detach().numpy(),batchsize)
            preds = model(dummy_input.cpu().detach().numpy(),batchsize)
            preds = softmax_stable(preds)
            indx = np.argmax(preds)
            FPS = 1/ (time.time() - start)
            print("FPS {}".format(FPS))
            # Map the predicted classes to the label names
            pred_class_names = kinetics_id_to_classname[int(indx)]
            #print("Predicted labels: {}".format(pred_class_names))
            visualization_state(frame, pred_class_names, (15, 255, 255))
            avg_fps.append(FPS)
            clip.pop(0)
        img = cv2.resize(frame, (600, 320))
        out.write(frame)
        cv2.imshow("output", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
avg_fps = sum(avg_fps[1:]) / len(avg_fps[1:])
print("Average FPS: {}".format(avg_fps))

cap.release()
out.release()
cv2.destroyAllWindows()
