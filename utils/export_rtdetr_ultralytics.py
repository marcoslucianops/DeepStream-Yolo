import os
import sys
import argparse
import warnings
import onnx
import torch
import torch.nn as nn
from copy import deepcopy
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder


class DeepStreamOutput(nn.Module):
    def __init__(self, img_size):
        self.img_size = img_size
        super().__init__()

    def forward(self, x):
        boxes = x[:, :, :4]
        boxes[:, :, [0, 2]] *= self.img_size[1]
        boxes[:, :, [1, 3]] *= self.img_size[0]
        scores, classes = torch.max(x[:, :, 4:], 2, keepdim=True)
        classes = classes.float()
        return boxes, scores, classes


def suppress_warnings():
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)


def rtdetr_ultralytics_export(weights, device):
    model = RTDETR(weights)
    model = deepcopy(model.model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()
    for k, m in model.named_modules():
        if isinstance(m, (Detect, RTDETRDecoder)):
            m.dynamic = False
            m.export = True
            m.format = 'onnx'
        elif isinstance(m, C2f):
            m.forward = m.forward_split
    return model


def main(args):
    suppress_warnings()

    print('\nStarting: %s' % args.weights)

    print('Opening RT-DETR Ultralytics model\n')

    device = select_device('cpu')
    model = rtdetr_ultralytics_export(args.weights, device)

    if len(model.names.keys()) > 0:
        print('\nCreating labels.txt file')
        f = open('labels.txt', 'w')
        for name in model.names.values():
            f.write(name + '\n')
        f.close()

    img_size = args.size * 2 if len(args.size) == 1 else args.size

    model = nn.Sequential(model, DeepStreamOutput(img_size))

    onnx_input_im = torch.zeros(args.batch, 3, *img_size).to(device)
    onnx_output_file = os.path.basename(args.weights).split('.pt')[0] + '.onnx'

    dynamic_axes = {
        'input': {
            0: 'batch'
        },
        'boxes': {
            0: 'batch'
        },
        'scores': {
            0: 'batch'
        },
        'classes': {
            0: 'batch'
        }
    }

    print('\nExporting the model to ONNX')
    torch.onnx.export(model, onnx_input_im, onnx_output_file, verbose=False, opset_version=args.opset,
                      do_constant_folding=True, input_names=['input'], output_names=['boxes', 'scores', 'classes'],
                      dynamic_axes=dynamic_axes if args.dynamic else None)

    if args.simplify:
        print('Simplifying the ONNX model')
        import onnxsim
        model_onnx = onnx.load(onnx_output_file)
        model_onnx, _ = onnxsim.simplify(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print('Done: %s\n' % onnx_output_file)


def parse_args():
    parser = argparse.ArgumentParser(description='DeepStream RT-DETR Ultralytics conversion')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--opset', type=int, default=16, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Static batch-size')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and static batch-size at same time')
    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
