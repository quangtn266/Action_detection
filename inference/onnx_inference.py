import onnxruntime
import onnx
import cv2
import numpy as np
import os
from utils.utils import post_process, multi_hot_vis
from config import build_model_config, build_dataset_config
from dataset.transforms import BaseTransform
from PIL import Image

import time
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Loading .onnx file of a pytorchvideo for a inference')
    parser.add_argument('--onnx-file', type=str, help='.onnx file', default="./backup/onnx_files/yowo_ava_v2.2_20.6.onnx")
    parser.add_argument('--output', type=str, help='output video', default="./backup/tmp.mp4")
    parser.add_argument('--video', type=str, help='web camera or a video', default="./backup/action_recognition.mov")
    parser.add_argument('--dataset', type=str, help='dataset classes', default="./config/ava_categories_ratio.json")
    parser.add_argument('--version', default='yowo', type=str, choices=['yowo', 'yowo_nano'], help='build YOWO')
    parser.add_argument('-vs', '--vis_thresh', default=0.3, type=float, help='threshold for visualization')
    parser.add_argument('--topk', default=40, type=int, help='NMS threshold')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    import json

    with open(args.dataset, "r") as f:
        classnames = json.load(f)

    # Create an id to label name mapping
    id_to_classname = {}
    for k, v in classnames.items():
        id_to_classname[v] = str(k).replace('"', "")

    if args.video != "None":
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)
    retraining = True
    version = args.version
    dataset = args.dataset
    m_cfg = build_model_config(version)
    d_cfg = build_dataset_config("ava_v2.2")

    # transform
    basetransform = BaseTransform(
        img_size=d_cfg['test_size'],
        pixel_mean=d_cfg['pixel_mean'],
        pixel_std=d_cfg['pixel_std']
    )

    sessions = onnxruntime.InferenceSession(args.onnx_file)
    sessions.get_modelmeta()
    first_input_name = sessions.get_inputs()[0].name
    first_output_name = sessions.get_outputs()[0].name

    clip = []
    avg_fps = []
    pred_class_names = " "
    video_clip = []

    videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(args.output, fourcc, 10.0, (videoWidth, videoHeight), True)

    while retraining:
        retraining, frame = cap.read()
        if not retraining and frame is None:
            continue

        # orig size
        orig_h, orig_w = frame.shape[:2]
        if retraining:
            # to RGB
            frame_rgb = frame[..., (2, 1, 0)]

            # to PIL image
            frame_pil = Image.fromarray(frame_rgb.astype(np.uint8))

            # prepare
            if len(video_clip) <= 0:
                for _ in range(d_cfg['len_clip']):
                    video_clip.append(frame_pil)

            video_clip.append(frame_pil)


            # orig size
            orig_h, orig_w = frame.shape[:2]

            # transform
            x, _ = basetransform(video_clip)
            # List [T, 3, H, W] -> [3, T, H, W]
            x = torch.stack(x, dim=1)
            x = x.unsqueeze(0).to(args.device)  # [B, 3, T, H, W], B=1

            start = time.time()
            preds = sessions.run([first_output_name], {first_input_name: x.cpu().detach().numpy()})
            preds_list = []
            if d_cfg['multi_hot']:
                batch_bboxes = post_process(preds, m_cfg['anchor_size']['ava_v2.2'], d_cfg['multi_hot'], d_cfg['conf_thresh'], \
                               d_cfg['test_size'], d_cfg['nms_thresh'], args.topk, args.device, num_classes=80)
            else:
                batch_scores, batch_labels, batch_bboxes = post_process(preds, m_cfg['anchor_size']['ava_v2.2'], d_cfg['multi_hot'], d_cfg['conf_thresh'], \
                               d_cfg['test_size'], d_cfg['nms_thresh'], args.topk, args.device, num_classes=80)

            for bi in range(len(batch_bboxes)):
                bboxes = batch_bboxes[bi]

                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox[:4]
                    det_conf = float(bbox[4])
                    cls_out = [det_conf * cls_conf for cls_conf in bbox[5:]]
                    preds_list.append([[x1, y1, x2, y2], cls_out])
                frame = multi_hot_vis(
                    args=args,
                    frame=frame,
                    out_bboxes=bboxes,
                    orig_w=orig_w,
                    orig_h=orig_h,
                    class_names=d_cfg['label_map'],
                    act_pose=False
                )
            FPS = 1 / (time.time() - start)
            print("FPS {}".format(FPS))
            avg_fps.append(FPS)
            video_clip.pop(0)
        img = cv2.resize(frame, (800, 600))
        out.write(frame)
        cv2.imshow("output", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

avg_fps = sum(avg_fps[1:]) / len(avg_fps[1:])
print("Average FPS: {}".format(avg_fps))

cap.release()
out.release()
cv2.destroyAllWindows()