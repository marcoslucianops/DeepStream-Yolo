import os
import sys
import argparse
import warnings
import onnx
import torch
import torch.nn as nn
from super_gradients.training import models


class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        boxes = x[0]
        scores, classes = torch.max(x[1], 2, keepdim=True)
        classes = classes.float()
        return boxes, scores, classes


def suppress_warnings():
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)


def yolonas_export(model_name, weights, num_classes, size):
    img_size = size * 2 if len(size) == 1 else size
    model = models.get(model_name, num_classes=num_classes, checkpoint_path=weights)
    model.eval()
    model.prep_model_for_conversion(input_size=[1, 3, *img_size])
    return model


def main(args):
    suppress_warnings()

    print('\nStarting: %s' % args.weights)

    print('Opening YOLO-NAS model\n')

    device = torch.device('cpu')
    model = yolonas_export(args.model, args.weights, args.classes, args.size)

    model = nn.Sequential(model, DeepStreamOutput())

    img_size = args.size * 2 if len(args.size) == 1 else args.size

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
    parser = argparse.ArgumentParser(description='DeepStream YOLO-NAS conversion')
    parser.add_argument('-m', '--model', required=True, help='Model name (required)')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pth) file path (required)')
    parser.add_argument('-n', '--classes', type=int, default=80, help='Number of trained classes (default 80)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--opset', type=int, default=14, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Implicit batch-size')
    args = parser.parse_args()
    if args.model == '':
        raise SystemExit('Invalid model name')
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and implicit batch-size at same time')
    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
