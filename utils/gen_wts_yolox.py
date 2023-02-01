import argparse
import os
import struct
import torch
from yolox.exp import get_exp


class Layers(object):
    def __init__(self, size, fw, fc):
        self.blocks = [0 for _ in range(300)]
        self.current = -1

        self.width = size[0] if len(size) == 1 else size[1]
        self.height = size[0]

        self.backbone_outs = []
        self.fpn_feats = []
        self.pan_feats = []
        self.yolo_head = []

        self.fw = fw
        self.fc = fc
        self.wc = 0

        self.net()

    def Conv(self, child):
        self.current += 1

        if child._get_name() == 'DWConv':
            self.convolutional(child.dconv)
            self.convolutional(child.pconv)
        else:
            self.convolutional(child)

    def Focus(self, child):
        self.current += 1

        self.reorg()
        self.convolutional(child.conv)

    def BaseConv(self, child, stage='', act=None):
        self.current += 1

        self.convolutional(child, act=act)
        if stage == 'fpn':
            self.fpn_feats.append(self.current)

    def CSPLayer(self, child, stage=''):
        self.current += 1

        self.convolutional(child.conv2)
        self.route('-2')
        self.convolutional(child.conv1)
        idx = -3
        for m in child.m:
            if m.use_add:
                self.convolutional(m.conv1)
                if m.conv2._get_name() == 'DWConv':
                    self.convolutional(m.conv2.dconv)
                    self.convolutional(m.conv2.pconv)
                    self.shortcut(-4)
                    idx -= 4
                else:
                    self.convolutional(m.conv2)
                    self.shortcut(-3)
                    idx -= 3
            else:
                self.convolutional(m.conv1)
                if m.conv2._get_name() == 'DWConv':
                    self.convolutional(m.conv2.dconv)
                    self.convolutional(m.conv2.pconv)
                    idx -= 3
                else:
                    self.convolutional(m.conv2)
                    idx -= 2
        self.route('-1, %d' % idx)
        self.convolutional(child.conv3)
        if stage == 'backbone':
            self.backbone_outs.append(self.current)
        elif stage == 'pan':
            self.pan_feats.append(self.current)

    def SPPBottleneck(self, child):
        self.current += 1

        self.convolutional(child.conv1)
        self.maxpool(child.m[0])
        self.route('-2')
        self.maxpool(child.m[1])
        self.route('-4')
        self.maxpool(child.m[2])
        self.route('-6, -5, -3, -1')
        self.convolutional(child.conv2)

    def Upsample(self, child):
        self.current += 1

        self.upsample(child)

    def Concat(self, route):
        self.current += 1

        r = self.get_route(route)
        self.route('-1, %d' % r)

    def Route(self, route):
        self.current += 1

        if route > 0:
            r = self.get_route(route)
            self.route('%d' % r)
        else:
            self.route('%d' % route)

    def RouteShuffleOut(self, route):
        self.current += 1

        self.route(route)
        self.shuffle(reshape=['c', 'hw'])
        self.yolo_head.append(self.current)

    def Detect(self, strides):
        self.current += 1

        routes = self.yolo_head[::-1]

        for i, route in enumerate(routes):
            routes[i] = self.get_route(route)
        self.route(str(routes)[1:-1], axis=1)
        self.shuffle(transpose1=[1, 0])
        self.yolo(strides)

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
            act = act if act is not None else 'linear'
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

    def yolo(self, strides):
        self.blocks[self.current] += 1

        self.fc.write('\n[detect_x]\n' +
                      'strides=%s\n' % str(strides)[1:-1])

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
    parser = argparse.ArgumentParser(description='PyTorch YOLOX conversion')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pth) file path (required)')
    parser.add_argument('-e', '--exp', required=True, help='Input exp (.py) file path (required)')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if not os.path.isfile(args.exp):
        raise SystemExit('Invalid exp file')
    return args.weights, args.exp


pth_file, exp_file = parse_args()

exp = get_exp(exp_file)
model = exp.get_model()
model.load_state_dict(torch.load(pth_file, map_location='cpu')['model'])
model.to('cpu').eval()

model_name = exp.exp_name
inference_size = (exp.input_size[1], exp.input_size[0])

backbone = model.backbone._get_name()
head = model.head._get_name()

wts_file = model_name + '.wts' if 'yolox' in model_name else 'yolox_' + model_name + '.wts'
cfg_file = model_name + '.cfg' if 'yolox' in model_name else 'yolox_' + model_name + '.cfg'

with open(wts_file, 'w') as fw, open(cfg_file, 'w') as fc:
    layers = Layers(inference_size, fw, fc)

    if backbone == 'YOLOPAFPN':
        layers.fc.write('\n# YOLOPAFPN\n')

        layers.Focus(model.backbone.backbone.stem)
        layers.Conv(model.backbone.backbone.dark2[0])
        layers.CSPLayer(model.backbone.backbone.dark2[1])
        layers.Conv(model.backbone.backbone.dark3[0])
        layers.CSPLayer(model.backbone.backbone.dark3[1], 'backbone')
        layers.Conv(model.backbone.backbone.dark4[0])
        layers.CSPLayer(model.backbone.backbone.dark4[1], 'backbone')
        layers.Conv(model.backbone.backbone.dark5[0])
        layers.SPPBottleneck(model.backbone.backbone.dark5[1])
        layers.CSPLayer(model.backbone.backbone.dark5[2], 'backbone')
        layers.BaseConv(model.backbone.lateral_conv0, 'fpn')
        layers.Upsample(model.backbone.upsample)
        layers.Concat(layers.backbone_outs[1])
        layers.CSPLayer(model.backbone.C3_p4)
        layers.BaseConv(model.backbone.reduce_conv1, 'fpn')
        layers.Upsample(model.backbone.upsample)
        layers.Concat(layers.backbone_outs[0])
        layers.CSPLayer(model.backbone.C3_p3, 'pan')
        layers.Conv(model.backbone.bu_conv2)
        layers.Concat(layers.fpn_feats[1])
        layers.CSPLayer(model.backbone.C3_n3, 'pan')
        layers.Conv(model.backbone.bu_conv1)
        layers.Concat(layers.fpn_feats[0])
        layers.CSPLayer(model.backbone.C3_n4, 'pan')
        layers.pan_feats = layers.pan_feats[::-1]
    else:
        raise SystemExit('Model not supported')

    if head == 'YOLOXHead':
        layers.fc.write('\n# YOLOXHead\n')

        for i, feat in enumerate(layers.pan_feats):
            idx = len(layers.pan_feats) - i - 1
            dw = True if model.head.cls_convs[idx][0]._get_name() == 'DWConv' else False
            if i > 0:
                layers.Route(feat)
            layers.BaseConv(model.head.stems[idx])
            layers.Conv(model.head.cls_convs[idx][0])
            layers.Conv(model.head.cls_convs[idx][1])
            layers.BaseConv(model.head.cls_preds[idx], act='sigmoid')
            if dw:
                layers.Route(-6)
            else:
                layers.Route(-4)
            layers.Conv(model.head.reg_convs[idx][0])
            layers.Conv(model.head.reg_convs[idx][1])
            layers.BaseConv(model.head.obj_preds[idx], act='sigmoid')
            layers.Route(-2)
            layers.BaseConv(model.head.reg_preds[idx])
            if dw:
                layers.RouteShuffleOut('-1, -3, -9')
            else:
                layers.RouteShuffleOut('-1, -3, -7')
        layers.Detect(model.head.strides)

    else:
        raise SystemExit('Model not supported')

os.system('echo "%d" | cat - %s > temp && mv temp %s' % (layers.wc, wts_file, wts_file))
