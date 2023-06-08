import os
import sys
import argparse
import warnings
import onnx
import torch
import torch.nn as nn
import models
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.activations import Hardswish, SiLU


class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        boxes = x[:, :, :4]
        objectness = x[:, :, 4:5]
        scores, classes = torch.max(x[:, :, 5:], 2, keepdim=True)
        scores *= objectness
        classes = classes.float()
        return boxes, scores, classes


def suppress_warnings():
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)


def yolov7_export(weights, device):
    model = attempt_load(weights, map_location=device)
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()
        if isinstance(m, models.common.Conv):
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    model.model[-1].export = False
    model.model[-1].concat = True
    model.eval()
    return model


def main(args):
    suppress_warnings()

    print('\nStarting: %s' % args.weights)

    print('Opening YOLOv7 model\n')

    device = select_device('cpu')
    model = yolov7_export(args.weights, device)

    if len(model.names) > 0:
        print('\nCreating labels.txt file')
        f = open('labels.txt', 'w')
        for name in model.names:
            f.write(name + '\n')
        f.close()

    model = nn.Sequential(model, DeepStreamOutput())

    img_size = args.size * 2 if len(args.size) == 1 else args.size

    if img_size == [640, 640] and args.p6:
        img_size = [1280] * 2

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
    parser = argparse.ArgumentParser(description='DeepStream YOLOv7 conversion')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--p6', action='store_true', help='P6 model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Implicit batch-size')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and implicit batch-size at same time')
    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
