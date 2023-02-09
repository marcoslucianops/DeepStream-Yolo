import argparse
import os
import struct
import torch
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.utils.tal import make_anchors


class Layers(object):
    def __init__(self, n, size, fw, fc):
        self.blocks = [0 for _ in range(n)]
        self.current = -1

        self.width = size[0] if len(size) == 1 else size[1]
        self.height = size[0]

        self.fw = fw
        self.fc = fc
        self.wc = 0

        self.net()

    def Conv(self, child):
        self.current = child.i
        self.fc.write('\n# Conv\n')

        self.convolutional(child)

    def C2f(self, child):
        self.current = child.i
        self.fc.write('\n# C2f\n')

        self.convolutional(child.cv1)
        self.c2f(child.m)
        self.convolutional(child.cv2)

    def SPPF(self, child):
        self.current = child.i
        self.fc.write('\n# SPPF\n')

        self.convolutional(child.cv1)
        self.maxpool(child.m)
        self.maxpool(child.m)
        self.maxpool(child.m)
        self.route('-4, -3, -2, -1')
        self.convolutional(child.cv2)

    def Upsample(self, child):
        self.current = child.i
        self.fc.write('\n# Upsample\n')

        self.upsample(child)

    def Concat(self, child):
        self.current = child.i
        self.fc.write('\n# Concat\n')

        r = []
        for i in range(1, len(child.f)):
            r.append(self.get_route(child.f[i]))
        self.route('-1, %s' % str(r)[1:-1])

    def Detect(self, child):
        self.current = child.i
        self.fc.write('\n# Detect\n')

        output_idxs = [0 for _ in range(child.nl)]
        for i in range(child.nl):
            r = self.get_route(child.f[i])
            self.route('%d' % r)
            for j in range(len(child.cv3[i])):
                self.convolutional(child.cv3[i][j])
            self.route('%d' % (-1 - len(child.cv3[i])))
            for j in range(len(child.cv2[i])):
                self.convolutional(child.cv2[i][j])
            self.route('-1, %d' % (-2 - len(child.cv2[i])))
            self.shuffle(reshape=[child.no, -1])
            output_idxs[i] = (-1 + i * (-4 - len(child.cv3[i]) - len(child.cv2[i])))
        self.route('%s' % str(output_idxs[::-1])[1:-1], axis=1)
        self.yolo(child)

    def net(self):
        self.fc.write('[net]\n' +
                      'width=%d\n' % self.width +
                      'height=%d\n' % self.height +
                      'channels=3\n' +
                      'letter_box=1\n')

    def convolutional(self, cv, act=None, detect=False):
        self.blocks[self.current] += 1

        self.get_state_dict(cv.state_dict())

        if cv._get_name() == 'Conv2d':
            filters = cv.out_channels
            size = cv.kernel_size
            stride = cv.stride
            pad = cv.padding
            groups = cv.groups
            bias = cv.bias
            bn = False
            act = 'linear' if not detect else 'logistic'
        else:
            filters = cv.conv.out_channels
            size = cv.conv.kernel_size
            stride = cv.conv.stride
            pad = cv.conv.padding
            groups = cv.conv.groups
            bias = cv.conv.bias
            bn = True if hasattr(cv, 'bn') else False
            if act is None:
                act = self.get_activation(cv.act._get_name()) if hasattr(cv, 'act') else 'linear'

        b = 'batch_normalize=1\n' if bn is True else ''
        g = 'groups=%d\n' % groups if groups > 1 else ''
        w = 'bias=1\n' if bias is not None and bn is not False else 'bias=0\n' if bias is None and bn is False else ''

        self.fc.write('\n[convolutional]\n' +
                      b +
                      'filters=%d\n' % filters +
                      'size=%s\n' % self.get_value(size) +
                      'stride=%s\n' % self.get_value(stride) +
                      'pad=%s\n' % self.get_value(pad) +
                      g +
                      w +
                      'activation=%s\n' % act)

    def c2f(self, m):
        self.blocks[self.current] += 1

        for x in m:
            self.get_state_dict(x.state_dict())

        n = len(m)
        shortcut = 1 if m[0].add else 0
        filters = m[0].cv1.conv.out_channels
        size = m[0].cv1.conv.kernel_size
        stride = m[0].cv1.conv.stride
        pad = m[0].cv1.conv.padding
        groups = m[0].cv1.conv.groups
        bias = m[0].cv1.conv.bias
        bn = True if hasattr(m[0].cv1, 'bn') else False
        act = 'linear'
        if hasattr(m[0].cv1, 'act'):
            act = self.get_activation(m[0].cv1.act._get_name()) 

        b = 'batch_normalize=1\n' if bn is True else ''
        g = 'groups=%d\n' % groups if groups > 1 else ''
        w = 'bias=1\n' if bias is not None and bn is not False else 'bias=0\n' if bias is None and bn is False else ''

        self.fc.write('\n[c2f]\n' +
                      'n=%d\n' % n +
                      'shortcut=%d\n' % shortcut +
                      b +
                      'filters=%d\n' % filters +
                      'size=%s\n' % self.get_value(size) +
                      'stride=%s\n' % self.get_value(stride) +
                      'pad=%s\n' % self.get_value(pad) +
                      g +
                      w +
                      'activation=%s\n' % act)

    def route(self, layers, axis=0):
        self.blocks[self.current] += 1

        a = 'axis=%d\n' % axis if axis != 0 else ''

        self.fc.write('\n[route]\n' +
                      'layers=%s\n' % layers +
                      a)

    def shortcut(self, r, ew='add', act='linear'):
        self.blocks[self.current] += 1

        m = 'mode=mul\n' if ew == 'mul' else ''

        self.fc.write('\n[shortcut]\n' +
                      'from=%d\n' % r +
                      m +
                      'activation=%s\n' % act)

    def maxpool(self, m):
        self.blocks[self.current] += 1

        stride = m.stride
        size = m.kernel_size
        mode = m.ceil_mode

        m = 'maxpool_up' if mode else 'maxpool'

        self.fc.write('\n[%s]\n' % m +
                      'stride=%d\n' % stride +
                      'size=%d\n' % size)

    def upsample(self, child):
        self.blocks[self.current] += 1

        stride = child.scale_factor

        self.fc.write('\n[upsample]\n' +
                      'stride=%d\n' % stride)

    def shuffle(self, reshape=None, transpose1=None, transpose2=None):
        self.blocks[self.current] += 1

        r = 'reshape=%s\n' % ', '.join(str(x) for x in reshape) if reshape is not None else ''
        t1 = 'transpose1=%s\n' % ', '.join(str(x) for x in transpose1) if transpose1 is not None else ''
        t2 = 'transpose2=%s\n' % ', '.join(str(x) for x in transpose2) if transpose2 is not None else ''

        self.fc.write('\n[shuffle]\n' +
                      r +
                      t1 +
                      t2)

    def yolo(self, child):
        self.blocks[self.current] += 1

        self.fc.write('\n[detect_v8]\n' +
                      'num=%d\n' % (child.reg_max * 4) +
                      'classes=%d\n' % child.nc)

    def get_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if 'num_batches_tracked' not in k:
                vr = v.reshape(-1).numpy()
                self.fw.write('{} {} '.format(k, len(vr)))
                for vv in vr:
                    self.fw.write(' ')
                    self.fw.write(struct.pack('>f', float(vv)).hex())
                self.fw.write('\n')
                self.wc += 1

    def get_anchors(self, anchor_points, stride_tensor):
        vr = anchor_points.numpy()
        self.fw.write('{} {} '.format('anchor_points', len(vr)))
        for vv in vr:
            self.fw.write(' ')
            self.fw.write(struct.pack('>f', float(vv)).hex())
        self.fw.write('\n')
        self.wc += 1
        vr = stride_tensor.numpy()
        self.fw.write('{} {} '.format('stride_tensor', len(vr)))
        for vv in vr:
            self.fw.write(' ')
            self.fw.write(struct.pack('>f', float(vv)).hex())
        self.fw.write('\n')
        self.wc += 1

    def get_value(self, key):
        if type(key) == int:
            return key
        return key[0] if key[0] == key[1] else str(key)[1:-1]

    def get_route(self, n):
        r = 0
        for i, b in enumerate(self.blocks):
            if i <= n:
                r += b
            else:
                break
        return r - 1

    def get_activation(self, act):
        if act == 'Hardswish':
            return 'hardswish'
        elif act == 'LeakyReLU':
            return 'leaky'
        elif act == 'SiLU':
            return 'silu'
        return 'linear'


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch YOLOv8 conversion')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument(
        '-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    return args.weights, args.size


pt_file, inference_size = parse_args()

model_name = os.path.basename(pt_file).split('.pt')[0]
wts_file = model_name + '.wts' if 'yolov8' in model_name else 'yolov8_' + model_name + '.wts'
cfg_file = model_name + '.cfg' if 'yolov8' in model_name else 'yolov8_' + model_name + '.cfg'

device = select_device('cpu')
model = torch.load(pt_file, map_location=device)['model'].float()
model.to(device).eval()

if model.names and model.nc:
    with open("labels.txt", 'w') as fw:
        for i in range(model.nc):
            fw.write(model.names[i] + '\n')

with open(wts_file, 'w') as fw, open(cfg_file, 'w') as fc:
    layers = Layers(len(model.model), inference_size, fw, fc)

    for child in model.model.children():
        if child._get_name() == 'Conv':
            layers.Conv(child)
        elif child._get_name() == 'C2f':
            layers.C2f(child)
        elif child._get_name() == 'SPPF':
            layers.SPPF(child)
        elif child._get_name() == 'Upsample':
            layers.Upsample(child)
        elif child._get_name() == 'Concat':
            layers.Concat(child)
        elif child._get_name() == 'Detect':
            layers.Detect(child)
            x = []
            for stride in model.stride.tolist():
                x.append(torch.zeros([1, 1, int(layers.height / stride), int(layers.width / stride)], dtype=torch.float32))
            anchor_points, stride_tensor = (x.transpose(0, 1) for x in make_anchors(x, child.stride, 0.5))
            layers.get_anchors(anchor_points.reshape([-1]), stride_tensor.reshape([-1]))
        else:
            raise SystemExit('Model not supported')

os.system('echo "%d" | cat - %s > temp && mv temp %s' % (layers.wc, wts_file, wts_file))
