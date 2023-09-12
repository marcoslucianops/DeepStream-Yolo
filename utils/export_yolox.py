import os
import sys
import argparse
import warnings
import onnx
import torch
import torch.nn as nn
from yolox.exp import get_exp
from yolox.utils import replace_module
from yolox.models.network_blocks import SiLU


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


def yolox_export(weights, exp_file):
    exp = get_exp(exp_file)
    model = exp.get_model()
    ckpt = torch.load(weights, map_location='cpu')
    model.eval()
    if 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = True
    return model, exp


def main(args):
    suppress_warnings()

    print('\nStarting: %s' % args.weights)

    print('Opening YOLOX model')

    device = torch.device('cpu')
    model, exp = yolox_export(args.weights, args.exp)

    model = nn.Sequential(model, DeepStreamOutput())

    img_size = [exp.input_size[1], exp.input_size[0]]

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

    print('Exporting the model to ONNX')
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
    parser = argparse.ArgumentParser(description='DeepStream YOLOX conversion')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pth) file path (required)')
    parser.add_argument('-c', '--exp', required=True, help='Input exp (.py) file path (required)')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Static batch-size')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if not os.path.isfile(args.exp):
        raise SystemExit('Invalid exp file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and static batch-size at same time')
    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
