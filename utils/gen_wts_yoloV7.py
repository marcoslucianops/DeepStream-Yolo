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

    def ReOrg(self, child):
        self.current = child.i
        self.fc.write('\n# ReOrg\n')

        self.reorg()

    def Conv(self, child):
        self.current = child.i
        self.fc.write('\n# Conv\n')

        if child.f != -1:
            r = self.get_route(child.f)
            self.route('%d' % r)
        self.convolutional(child)

    def DownC(self, child):
        self.current = child.i
        self.fc.write('\n# DownC\n')

        self.maxpool(child.mp)
        self.convolutional(child.cv3)
        self.route('-3')
        self.convolutional(child.cv1)
        self.convolutional(child.cv2)
        self.route('-1, -4')

    def MP(self, child):
        self.current = child.i
        self.fc.write('\n# MP\n')

        self.maxpool(child.m)

    def SP(self, child):
        self.current = child.i
        self.fc.write('\n# SP\n')

        if child.f != -1:
            r = self.get_route(child.f)
            self.route('%d' % r)
        self.maxpool(child.m)

    def SPPCSPC(self, child):
        self.current = child.i
        self.fc.write('\n# SPPCSPC\n')

        self.convolutional(child.cv2)
        self.route('-2')
        self.convolutional(child.cv1)
        self.convolutional(child.cv3)
        self.convolutional(child.cv4)
        self.maxpool(child.m[0])
        self.route('-2')
        self.maxpool(child.m[1])
        self.route('-4')
        self.maxpool(child.m[2])
        self.route('-6, -5, -3, -1')
        self.convolutional(child.cv5)
        self.convolutional(child.cv6)
        self.route('-1, -13')
        self.convolutional(child.cv7)

    def RepConv(self, child):
        self.current = child.i
        self.fc.write('\n# RepConv\n')

        if child.f != -1:
            r = self.get_route(child.f)
            self.route('%d' % r)
        self.convolutional(child.rbr_1x1)
        self.route('-2')
        self.convolutional(child.rbr_dense)
        self.shortcut(-3, act=self.get_activation(child.act._get_name()))

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

    def Shortcut(self, child):
        self.current = child.i
        self.fc.write('\n# Shortcut\n')

        r = self.get_route(child.f[1])
        self.shortcut(r)

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
        elif cv._get_name() == 'Sequential':
            filters = cv[0].out_channels
            size = cv[0].kernel_size
            stride = cv[0].stride
            pad = cv[0].padding
            groups = cv[0].groups
            bias = cv[0].bias
            bn = True if cv[1]._get_name() == 'BatchNorm2d' else False
            act = 'linear'
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
        w = 'bias=0\n' if bias is None and bn is False else ''

        self.fc.write('\n[convolutional]\n' +
                      b +
                      'filters=%d\n' % filters +
                      'size=%s\n' % self.get_value(size) +
                      'stride=%s\n' % self.get_value(stride) +
                      'pad=%s\n' % self.get_value(pad) +
                      g +
                      w +
                      'activation=%s\n' % act)

    def route(self, layers):
        self.blocks[self.current] += 1

        self.fc.write('\n[route]\n' +
                      'layers=%s\n' % layers)

    def shortcut(self, r, act='linear'):
        self.blocks[self.current] += 1

        self.fc.write('\n[shortcut]\n' +
                      'from=%d\n' % r +
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
    parser = argparse.ArgumentParser(description='PyTorch YOLOv7 conversion')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument(
        '-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    return args.weights, args.size


pt_file, inference_size = parse_args()

model_name = os.path.basename(pt_file).split('.pt')[0]
wts_file = model_name + '.wts' if 'yolov7' in model_name else 'yolov7_' + model_name + '.wts'
cfg_file = model_name + '.cfg' if 'yolov7' in model_name else 'yolov7_' + model_name + '.cfg'

device = select_device('cpu')
model = torch.load(pt_file, map_location=device)
model = model['ema' if model.get('ema') else 'model'].float()

anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]
delattr(model.model[-1], 'anchor_grid')
model.model[-1].register_buffer('anchor_grid', anchor_grid)

model.to(device).eval()

with open(wts_file, 'w') as fw, open(cfg_file, 'w') as fc:
    layers = Layers(len(model.model), inference_size, fw, fc)

    for child in model.model.children():
        if child._get_name() == 'ReOrg':
            layers.ReOrg(child)
        elif child._get_name() == 'Conv':
            layers.Conv(child)
        elif child._get_name() == 'DownC':
            layers.DownC(child)
        elif child._get_name() == 'MP':
            layers.MP(child)
        elif child._get_name() == 'SP':
            layers.SP(child)
        elif child._get_name() == 'SPPCSPC':
            layers.SPPCSPC(child)
        elif child._get_name() == 'RepConv':
            layers.RepConv(child)
        elif child._get_name() == 'Upsample':
            layers.Upsample(child)
        elif child._get_name() == 'Concat':
            layers.Concat(child)
        elif child._get_name() == 'Shortcut':
            layers.Shortcut(child)
        elif child._get_name() == 'Detect':
            layers.Detect(child)
        else:
            raise SystemExit('Model not supported')

os.system('echo "%d" | cat - %s > temp && mv temp %s' % (layers.wc, wts_file, wts_file))
