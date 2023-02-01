import argparse
import os
import struct
import torch
from utils.torch_utils import select_device


class Layers(object):
    def __init__(self, n, size, fw, fc):
        self.blocks = [0 for _ in range(n)]
        self.current = 0

        self.width = size[0] if len(size) == 1 else size[1]
        self.height = size[0]

        self.num = 0
        self.nc = 0
        self.anchors = ''
        self.masks = []

        self.fw = fw
        self.fc = fc
        self.wc = 0

        self.net()

    def Focus(self, child):
        self.current = child.i
        self.fc.write('\n# Focus\n')

        self.reorg()
        self.convolutional(child.conv)

    def Conv(self, child):
        self.current = child.i
        self.fc.write('\n# Conv\n')

        self.convolutional(child)

    def BottleneckCSP(self, child):
        self.current = child.i
        self.fc.write('\n# BottleneckCSP\n')

        self.convolutional(child.cv2)
        self.route('-2')
        self.convolutional(child.cv1)
        idx = -3
        for m in child.m:
            if m.add:
                self.convolutional(m.cv1)
                self.convolutional(m.cv2)
                self.shortcut(-3)
                idx -= 3
            else:
                self.convolutional(m.cv1)
                self.convolutional(m.cv2)
                idx -= 2
        self.convolutional(child.cv3)
        self.route('-1, %d' % (idx - 1))
        self.batchnorm(child.bn, child.act)
        self.convolutional(child.cv4)

    def C3(self, child):
        self.current = child.i
        self.fc.write('\n# C3\n')

        self.convolutional(child.cv2)
        self.route('-2')
        self.convolutional(child.cv1)
        idx = -3
        for m in child.m:
            if m.add:
                self.convolutional(m.cv1)
                self.convolutional(m.cv2)
                self.shortcut(-3)
                idx -= 3
            else:
                self.convolutional(m.cv1)
                self.convolutional(m.cv2)
                idx -= 2
        self.route('-1, %d' % idx)
        self.convolutional(child.cv3)

    def SPP(self, child):
        self.current = child.i
        self.fc.write('\n# SPP\n')

        self.convolutional(child.cv1)
        self.maxpool(child.m[0])
        self.route('-2')
        self.maxpool(child.m[1])
        self.route('-4')
        self.maxpool(child.m[2])
        self.route('-6, -5, -3, -1')
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

        self.get_anchors(child.state_dict(), child.m[0].out_channels)

        for i, m in enumerate(child.m):
            r = self.get_route(child.f[i])
            self.route('%d' % r)
            self.convolutional(m, detect=True)
            self.yolo(i)

    def net(self):
        self.fc.write('[net]\n' +
                      'width=%d\n' % self.width +
                      'height=%d\n' % self.height +
                      'channels=3\n' +
                      'letter_box=1\n')

    def CBH(self, child):
        self.current = child.i
        self.fc.write('\n# CBH\n')

        self.convolutional(child.conv, act='hardswish')

    def LC_Block(self, child):
        self.current = child.i
        self.fc.write('\n# LC_Block\n')

        self.convolutional(child.dw_conv, act='hardswish')
        if child.use_se:
            self.avgpool()
            self.convolutional(child.se.conv1, act='relu')
            self.convolutional(child.se.conv2, act='silu')
            self.shortcut(-4, ew='mul')
        self.convolutional(child.pw_conv, act='hardswish')

    def Dense(self, child):
        self.current = child.i
        self.fc.write('\n# Dense\n')

        self.convolutional(child.dense_conv, act='hardswish')

    def reorg(self):
        self.blocks[self.current] += 1

        self.fc.write('\n[reorg]\n')

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

    def batchnorm(self, bn, act):
        self.blocks[self.current] += 1

        self.get_state_dict(bn.state_dict())

        filters = bn.num_features
        act = self.get_activation(act._get_name())

        self.fc.write('\n[batchnorm]\n' +
                      'filters=%d\n' % filters +
                      'activation=%s\n' % act)

    def route(self, layers):
        self.blocks[self.current] += 1

        self.fc.write('\n[route]\n' +
                      'layers=%s\n' % layers)

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

    def avgpool(self):
        self.blocks[self.current] += 1

        self.fc.write('\n[avgpool]\n')

    def yolo(self, i):
        self.blocks[self.current] += 1

        self.fc.write('\n[yolo]\n' +
                      'mask=%s\n' % self.masks[i] +
                      'anchors=%s\n' % self.anchors +
                      'classes=%d\n' % self.nc +
                      'num=%d\n' % self.num +
                      'scale_x_y=2.0\n' +
                      'new_coords=1\n')

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

    def get_anchors(self, state_dict, out_channels):
        anchor_grid = state_dict['anchor_grid']
        aa = anchor_grid.reshape(-1).tolist()
        am = anchor_grid.tolist()

        self.num = (len(aa) / 2)
        self.nc = int((out_channels / (self.num / len(am))) - 5)
        self.anchors = str(aa)[1:-1]

        n = 0
        for m in am:
            mask = []
            for _ in range(len(m)):
                mask.append(n)
                n += 1
            self.masks.append(str(mask)[1:-1])

    def get_value(self, key):
        if type(key) == int:
            return key
        return key[0] if key[0] == key[1] else str(key)[1:-1]

    def get_route(self, n):
        r = 0
        if n < 0:
            for i, b in enumerate(self.blocks[self.current-1::-1]):
                if i < abs(n) - 1:
                    r -= b
                else:
                    break
        else:
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
    parser = argparse.ArgumentParser(description='PyTorch YOLOv5 conversion')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument(
        '-s', '--size', nargs='+', type=int, help='Inference size [H,W] (default [640])')
    parser.add_argument("--p6", action="store_true", help="P6 model")
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if not args.size:
        args.size = [1280] if args.p6 else [640]
    return args.weights, args.size


pt_file, inference_size = parse_args()

model_name = os.path.basename(pt_file).split('.pt')[0]
wts_file = model_name + '.wts' if 'yolov5' in model_name else 'yolov5_' + model_name + '.wts'
cfg_file = model_name + '.cfg' if 'yolov5' in model_name else 'yolov5_' + model_name + '.cfg'

device = select_device('cpu')
model = torch.load(pt_file, map_location=device)['model'].float()

anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]
delattr(model.model[-1], 'anchor_grid')
model.model[-1].register_buffer('anchor_grid', anchor_grid)

model.to(device).eval()

with open(wts_file, 'w') as fw, open(cfg_file, 'w') as fc:
    layers = Layers(len(model.model), inference_size, fw, fc)

    for child in model.model.children():
        if child._get_name() == 'Focus':
            layers.Focus(child)
        elif child._get_name() == 'Conv':
            layers.Conv(child)
        elif child._get_name() == 'BottleneckCSP':
            layers.BottleneckCSP(child)
        elif child._get_name() == 'C3':
            layers.C3(child)
        elif child._get_name() == 'SPP':
            layers.SPP(child)
        elif child._get_name() == 'SPPF':
            layers.SPPF(child)
        elif child._get_name() == 'Upsample':
            layers.Upsample(child)
        elif child._get_name() == 'Concat':
            layers.Concat(child)
        elif child._get_name() == 'Detect':
            layers.Detect(child)
        elif child._get_name() == 'CBH':
            layers.CBH(child)
        elif child._get_name() == 'LC_Block':
            layers.LC_Block(child)
        elif child._get_name() == 'Dense':
            layers.Dense(child)
        else:
            raise SystemExit('Model not supported')

os.system('echo "%d" | cat - %s > temp && mv temp %s' % (layers.wc, wts_file, wts_file))
