#### mmaction2 for R&D
1. Research onnx conversion for action- detection "SpatioTemporal Action Detection".

```
python tools/deployment/export_onnx_stdet.py ./configs/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb.py \ 
   	./checkpoints/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb_20220906-0dae1a90.pth  \
	--num_frames 8 --device cpu --output_file ./checkpoints/stdet.onnx
```

2. Inference onnx file. Currently, only slowonly version can run with the inference code.
```
python demo/demo_spatiotemporal_det_onnx.py demo/demo.mp4 demo/demo_spatiotemporal_det.mp4 \
    --config configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py \
    --onnx-file slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.onnx \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --action-score-thr 0.5 \
    --label-map tools/data/ava/label_map.txt \
    --predict-stepsize 8 \
    --output-stepsize 4 \
    --output-fps 6
```
