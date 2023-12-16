import torch
import argparse
from utils.misc import load_weight
from models.detector import build_model
from config import build_model_config, build_dataset_config

torch.set_default_tensor_type('torch.FloatTensor')

def sinle_input_model(model, model_name, show, opset_version, device):
    # Initialize a dummy input tensor.
    input = torch.randn(1, 3, 17, 224, 224, device=device)
    # Process slow and fast pathway for pretrained models (2 inputs)
    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
    output_file = "{}".format(model_name)
    print("Initialize the parameters of YOWO model")

    # Export onnx files (pre-trained models with one tensor)
    with torch.no_grad():
        torch.onnx.export(
            model, (input, ),
            output_file,
            input_names=["input"],
            output_names=['output'],
            export_params=True,
            do_constant_folding=True,
            keep_initializers_as_inputs=False,
            verbose=show,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes)

    return print('Successfully exported ONNX model {}'.format(output_file))

def parse_args():
    parser = argparse.ArgumentParser(description='Convert a pytorchvideo model to ONNX')
    parser.add_argument('--version', default='yowo', type=str, choices=['yowo', 'yowo_nano'], help='build YOWO')
    parser.add_argument('--output', type=str, help='single input model or multiple input model', default="onnx_files/tmp.onnx")
    parser.add_argument('--dataset', default='ava_v2.2', help='ava_v2.2')
    parser.add_argument('--weight', default=None, type=str, help='Trained state_dict file path to open')
    parser.add_argument('--show', type=str, help='show onnx graph and segmentation results', default=True)
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument('--cuda', type=str, default='cpu')
    parser.add_argument('--topk', default=40, type=int, help='NMS threshold')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # cuda
    if args.cuda == "cuda":
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    version = args.version
    dataset = args.dataset
    m_cfg = build_model_config(version)
    d_cfg = build_dataset_config("ava_v2.2")

    num_classes = d_cfg['valid_num_classes']

    # build model
    model = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device,
        num_classes=num_classes,
        trainable=False
    )

    # load trained weight
    model = load_weight(model=model, path_to_ckpt=args.weight)

    # to eval
    model = model.to(device).eval()
    sinle_input_model(model, args.output, args.show, args.opset_version, args.cuda)
