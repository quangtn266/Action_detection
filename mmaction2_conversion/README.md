#### mmaction2 for R&D
1. Research onnx conversion for action- detection "SpatioTemporal Action Detection".

```
python tools/deployment/export_onnx_stdet.py ./configs/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb.py \ 
   	./checkpoints/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb_20220906-0dae1a90.pth  \
	--num_frames 8 --device cpu --output_file ./checkpoints/stdet.onnx
```
