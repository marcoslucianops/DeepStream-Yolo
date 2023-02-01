import argparse
import os
import struct
import torch
from yolov6.assigners.anchor_generator import generate_anchors


class Layers(object):
    def __init__(self, size, fw, fc):
        self.blocks = [0 for _ in range(300)]
        self.current = -1

        self.width = size[0] if len(size) == 1 else size[1]
        self.height = size[0]

        self.backbone_outs = []
        self.fpn_feats = []
        self.pan_feats = []
        self.yolo_head_cls = []
        self.yolo_head_reg = []

        self.fw = fw
        self.fc = fc
        self.wc = 0

        self.net()

    def BaseConv(self, child):
        self.current += 1

        if child._get_name() == 'RepVGGBlock':
            self.convolutional(child.rbr_reparam, act=self.get_activation(child.nonlinearity._get_name()))
        elif child._get_name() == 'ConvWrapper' or child._get_name() == 'SimConvWrapper':
            self.convolutional(child.block)
        else:
            raise SystemExit('Model not supported')

    def RepBlock(self, child, stage=''):
        self.current += 1

        if child.conv1._get_name() == 'RepVGGBlock':
            self.convolutional(child.conv1.rbr_reparam, act=self.get_activation(child.conv1.nonlinearity._get_name()))
            if child.block is not None:
                for m in child.block:
                    self.convolutional(m.rbr_reparam, act=self.get_activation(m.nonlinearity._get_name()))
        elif child.conv1._get_name() == 'ConvWrapper' or child.conv1._get_name() == 'SimConvWrapper':
            self.convolutional(child.conv1.block)
            if child.block is not None:
                for m in child.block:
                    self.convolutional(m.block)
        else:
            raise SystemExit('Model not supported')

        if stage == 'backbone':
            self.backbone_outs.append(self.current)
        elif stage == 'pan':
            self.pan_feats.append(self.current)

    def BepC3(self, child, stage=''):
        self.current += 1

        if child.concat is True:
            self.convolutional(child.cv2)
            self.route('-2')
        self.convolutional(child.cv1)
        idx = -3
        if child.m.conv1.conv1._get_name() == 'RepVGGBlock':
            self.convolutional(child.m.conv1.conv1.rbr_reparam,
                               act=self.get_activation(child.m.conv1.conv1.nonlinearity._get_name()))
            self.convolutional(child.m.conv1.conv2.rbr_reparam,
                               act=self.get_activation(child.m.conv1.conv2.nonlinearity._get_name()))
            idx -= 2
            if child.m.conv1.shortcut:
                self.shortcut(-3)
                idx -= 1
            if child.m.block is not None:
                for m in child.m.block:
                    self.convolutional(m.conv1.rbr_reparam, act=self.get_activation(m.conv1.nonlinearity._get_name()))
                    self.convolutional(m.conv2.rbr_reparam, act=self.get_activation(m.conv2.nonlinearity._get_name()))
                    idx -= 2
                    if m.shortcut:
                        self.shortcut(-3)
                        idx -= 1
        elif child.m.conv1.conv1._get_name() == 'ConvWrapper' or child.m.conv1.conv1._get_name() == 'SimConvWrapper':
            self.convolutional(child.m.conv1.conv1.block)
            self.convolutional(child.m.conv1.conv2.block)
            idx -= 2
            if child.m.conv1.shortcut:
                self.shortcut(-3)
                idx -= 1
            if child.m.block is not None:
                for m in child.m.block:
                    self.convolutional(m.conv1.block)
                    self.convolutional(m.conv2.block)
                    idx -= 2
                    if m.shortcut:
                        self.shortcut(-3)
                        idx -= 1
        else:
            raise SystemExit('Model not supported')

        if child.concat is True:
            self.route('-1, %d' % idx)
        self.convolutional(child.cv3)

        if stage == 'backbone':
            self.backbone_outs.append(self.current)
        elif stage == 'pan':
            self.pan_feats.append(self.current)

    def CSPSPPF(self, child):
        self.current += 1

        self.convolutional(child.cv2)
        self.route('-2')
        self.convolutional(child.cv1)
        self.convolutional(child.cv3)
        self.convolutional(child.cv4)
        self.maxpool(child.m)
        self.maxpool(child.m)
        self.maxpool(child.m)
        self.route('-4, -3, -2, -1')
        self.convolutional(child.cv5)
        self.convolutional(child.cv6)
        self.route('-11, -1')
        self.convolutional(child.cv7)
        self.backbone_outs.append(self.current)

    def SPPF(self, child):
        self.current += 1

        self.convolutional(child.cv1)
        self.maxpool(child.m)
        self.maxpool(child.m)
        self.maxpool(child.m)
        self.route('-4, -3, -2, -1')
        self.convolutional(child.cv2)
        self.backbone_outs.append(self.current)

    def SimConv(self, child, stage=''):
        self.current += 1

        self.convolutional(child)
        if stage == 'fpn':
            self.fpn_feats.append(self.current)

    def BiFusion(self, child, idx):
        self.current += 1

        self.deconvolutional(child.upsample.upsample_transpose)
        r = self.get_route(self.backbone_outs[- idx -2])
        self.route('%d' % r)
        self.convolutional(child.cv1)
        r = self.get_route(self.backbone_outs[- idx -3])
        self.route('%d' % r)
        self.convolutional(child.cv2)
        self.convolutional(child.downsample)
        self.route('-6, -4, -1')
        self.convolutional(child.cv3)

    def Upsample(self, child):
        self.current += 1

        self.deconvolutional(child.upsample_transpose)

    def Conv(self, child, act=None):
        self.current += 1

        self.convolutional(child, act=act)

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

    def Shuffle(self, reshape=None, transpose1=None, transpose2=None, output=''):
        self.current += 1

        self.shuffle(reshape=reshape, transpose1=transpose1, transpose2=transpose2)
        if output == 'cls':
            self.yolo_head_cls.append(self.current)
        elif output == 'reg':
            self.yolo_head_reg.append(self.current)

    def SoftMax(self, axes):
        self.current += 1

        self.softmax(axes)

    def Detect(self, output):
        self.current += 1

        routes = self.yolo_head_cls if output == 'cls' else self.yolo_head_reg

        for i, route in enumerate(routes):
            routes[i] = self.get_route(route)
        self.route(str(routes)[1:-1], axis=-1)
        self.yolo(output)

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

    def deconvolutional(self, cv):
        self.blocks[self.current] += 1

        self.get_state_dict(cv.state_dict())

        filters = cv.out_channels
        size = cv.kernel_size
        stride = cv.stride
        pad = cv.padding
        groups = cv.groups
        bias = cv.bias

        g = 'groups=%d\n' % groups if groups > 1 else ''
        w = 'bias=0\n' if bias is None else ''

        self.fc.write('\n[deconvolutional]\n' +
                      'filters=%d\n' % filters +
                      'size=%s\n' % self.get_value(size) +
                      'stride=%s\n' % self.get_value(stride) +
                      'pad=%s\n' % self.get_value(pad) +
                      g +
                      w)

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

    def shuffle(self, reshape=None, transpose1=None, transpose2=None):
        self.blocks[self.current] += 1

        r = 'reshape=%s\n' % ', '.join(str(x) for x in reshape) if reshape is not None else ''
        t1 = 'transpose1=%s\n' % ', '.join(str(x) for x in transpose1) if transpose1 is not None else ''
        t2 = 'transpose2=%s\n' % ', '.join(str(x) for x in transpose2) if transpose2 is not None else ''

        self.fc.write('\n[shuffle]\n' +
                      r +
                      t1 +
                      t2)

    def softmax(self, axes):
        self.blocks[self.current] += 1

        self.fc.write('\n[softmax]\n' +
                      'axes=%d\n' % axes)

    def yolo(self, output):
        self.blocks[self.current] += 1

        self.fc.write('\n[%s]\n' % output)

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
        elif act == 'ReLU':
            return 'relu'
        return 'linear'


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch YOLOv6 conversion')
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
wts_file = model_name + '.wts' if 'yolov6' in model_name else 'yolov6_' + model_name + '.wts'
cfg_file = model_name + '.cfg' if 'yolov6' in model_name else 'yolov6_' + model_name + '.cfg'

model = torch.load(pt_file, map_location='cpu')['model'].float()
model.to('cpu').eval()

for layer in model.modules():
    if layer._get_name() == 'RepVGGBlock':
        layer.switch_to_deploy()

backbones = ['EfficientRep', 'CSPBepBackbone']
necks = ['RepBiFPANNeck', 'CSPRepBiFPANNeck', 'RepPANNeck', 'CSPRepPANNeck']
backbones_p6 = ['EfficientRep6', 'CSPBepBackbone_P6']
necks_p6 = ['RepBiFPANNeck6', 'CSPRepBiFPANNeck_P6', 'RepPANNeck6', 'CSPRepPANNeck_P6']

with open(wts_file, 'w') as fw, open(cfg_file, 'w') as fc:
    layers = Layers(inference_size, fw, fc)

    if model.backbone._get_name() in backbones:
        layers.fc.write('\n# %s\n' % model.backbone._get_name())

        if model.backbone._get_name() == 'EfficientRep':
            block1 = layers.RepBlock
        elif model.backbone._get_name() == 'CSPBepBackbone':
            block1 = layers.BepC3

        if model.backbone.ERBlock_5[2]._get_name() == 'CSPSPPF' or model.backbone.ERBlock_5[2]._get_name() == 'SimCSPSPPF':
            block2 = layers.CSPSPPF
        elif model.backbone.ERBlock_5[2]._get_name() == 'SPPF' or model.backbone.ERBlock_5[2]._get_name() == 'SimSPPF':
            block2 = layers.SPPF
        else:
            raise SystemExit('Model not supported')

        layers.BaseConv(model.backbone.stem)
        layers.BaseConv(model.backbone.ERBlock_2[0])
        block1(model.backbone.ERBlock_2[1], 'backbone' if hasattr(model.backbone, 'fuse_P2') and
               model.backbone.fuse_P2 else '')
        layers.BaseConv(model.backbone.ERBlock_3[0])
        block1(model.backbone.ERBlock_3[1], 'backbone')
        layers.BaseConv(model.backbone.ERBlock_4[0])
        block1(model.backbone.ERBlock_4[1], 'backbone')
        layers.BaseConv(model.backbone.ERBlock_5[0])
        block1(model.backbone.ERBlock_5[1])
        block2(model.backbone.ERBlock_5[2])

    elif model.backbone._get_name() in backbones_p6:
        layers.fc.write('\n# %s\n' % model.backbone._get_name())

        if model.backbone._get_name() == 'EfficientRep6':
            block1 = layers.RepBlock
        elif model.backbone._get_name() == 'CSPBepBackbone_P6':
            block1 = layers.BepC3

        if model.backbone.ERBlock_6[2]._get_name() == 'CSPSPPF' or model.backbone.ERBlock_6[2]._get_name() == 'SimCSPSPPF':
            block2 = layers.CSPSPPF
        elif model.backbone.ERBlock_6[2]._get_name() == 'SPPF' or model.backbone.ERBlock_6[2]._get_name() == 'SimSPPF':
            block2 = layers.SPPF
        else:
            raise SystemExit('Model not supported')

        layers.BaseConv(model.backbone.stem)
        layers.BaseConv(model.backbone.ERBlock_2[0])
        block1(model.backbone.ERBlock_2[1], 'backbone' if model.backbone._get_name() == 'CSPBepBackbone_P6' or
               (hasattr(model.backbone, 'fuse_P2') and model.backbone.fuse_P2) else '')
        layers.BaseConv(model.backbone.ERBlock_3[0])
        block1(model.backbone.ERBlock_3[1], 'backbone')
        layers.BaseConv(model.backbone.ERBlock_4[0])
        block1(model.backbone.ERBlock_4[1], 'backbone')
        layers.BaseConv(model.backbone.ERBlock_5[0])
        block1(model.backbone.ERBlock_5[1], 'backbone')
        layers.BaseConv(model.backbone.ERBlock_6[0])
        block1(model.backbone.ERBlock_6[1])
        block2(model.backbone.ERBlock_6[2])

    else:
        raise SystemExit('Model not supported')

    if model.neck._get_name() in necks:
        layers.fc.write('\n# %s\n' % model.neck._get_name())

        if model.neck._get_name() == 'RepBiFPANNeck' or model.neck._get_name() == 'RepPANNeck':
            block = layers.RepBlock
        elif model.neck._get_name() == 'CSPRepBiFPANNeck' or model.neck._get_name() == 'CSPRepPANNeck':
            block = layers.BepC3

        layers.SimConv(model.neck.reduce_layer0, 'fpn')
        if 'Bi' in model.neck._get_name():
            layers.BiFusion(model.neck.Bifusion0, 0)
        else:
            layers.Upsample(model.neck.upsample0)
            layers.Concat(layers.backbone_outs[-2])
        block(model.neck.Rep_p4)
        layers.SimConv(model.neck.reduce_layer1, 'fpn')
        if 'Bi' in model.neck._get_name():
            layers.BiFusion(model.neck.Bifusion1, 1)
        else:
            layers.Upsample(model.neck.upsample1)
            layers.Concat(layers.backbone_outs[-3])
        block(model.neck.Rep_p3, 'pan')
        layers.SimConv(model.neck.downsample2)
        layers.Concat(layers.fpn_feats[1])
        block(model.neck.Rep_n3, 'pan')
        layers.SimConv(model.neck.downsample1)
        layers.Concat(layers.fpn_feats[0])
        block(model.neck.Rep_n4, 'pan')
        layers.pan_feats = layers.pan_feats[::-1]

    elif model.neck._get_name() in necks_p6:
        layers.fc.write('\n# %s\n' % model.neck._get_name())

        if model.neck._get_name() == 'RepBiFPANNeck6' or model.neck._get_name() == 'RepPANNeck6':
            block = layers.RepBlock
        elif model.neck._get_name() == 'CSPRepBiFPANNeck_P6' or model.neck._get_name() == 'CSPRepPANNeck_P6':
            block = layers.BepC3

        layers.SimConv(model.neck.reduce_layer0, 'fpn')
        if 'Bi' in model.neck._get_name():
            layers.BiFusion(model.neck.Bifusion0, 0)
        else:
            layers.Upsample(model.neck.upsample0)
            layers.Concat(layers.backbone_outs[-2])
        block(model.neck.Rep_p5)
        layers.SimConv(model.neck.reduce_layer1, 'fpn')
        if 'Bi' in model.neck._get_name():
            layers.BiFusion(model.neck.Bifusion1, 1)
        else:
            layers.Upsample(model.neck.upsample1)
            layers.Concat(layers.backbone_outs[-3])
        block(model.neck.Rep_p4)
        layers.SimConv(model.neck.reduce_layer2, 'fpn')
        if 'Bi' in model.neck._get_name():
            layers.BiFusion(model.neck.Bifusion2, 2)
        else:
            layers.Upsample(model.neck.upsample2)
            layers.Concat(layers.backbone_outs[-4])
        block(model.neck.Rep_p3, 'pan')
        layers.SimConv(model.neck.downsample2)
        layers.Concat(layers.fpn_feats[2])
        block(model.neck.Rep_n4, 'pan')
        layers.SimConv(model.neck.downsample1)
        layers.Concat(layers.fpn_feats[1])
        block(model.neck.Rep_n5, 'pan')
        layers.SimConv(model.neck.downsample0)
        layers.Concat(layers.fpn_feats[0])
        block(model.neck.Rep_n6, 'pan')
        layers.pan_feats = layers.pan_feats[::-1]

    else:
        raise SystemExit('Model not supported')

    if model.detect._get_name() == 'Detect':
        layers.fc.write('\n# Detect\n')

        for i, feat in enumerate(layers.pan_feats):
            idx = len(layers.pan_feats) - i - 1
            if i > 0:
                layers.Route(feat)
            layers.Conv(model.detect.stems[idx])
            layers.Conv(model.detect.cls_convs[idx])
            layers.Conv(model.detect.cls_preds[idx], act='sigmoid')
            layers.Shuffle(reshape=[model.detect.nc, 'hw'], output='cls')
            layers.Route(-4)
            layers.Conv(model.detect.reg_convs[idx])
            layers.Conv(model.detect.reg_preds[idx])
            if model.detect.use_dfl:
                layers.Shuffle(reshape=[4, model.detect.reg_max + 1, 'hw'], transpose2=[1, 0, 2])
                layers.SoftMax(0)
                layers.Conv(model.detect.proj_conv)
                layers.Shuffle(reshape=['h', 'w'], output='reg')
            else:
                layers.Shuffle(reshape=[4, 'hw'], output='reg')
        layers.Detect('cls')
        layers.Detect('reg')

        x = []
        for stride in model.detect.stride.tolist()[::-1]:
            x.append(torch.zeros([1, 1, int(layers.height / stride), int(layers.width / stride)], dtype=torch.float32))
        anchor_points, stride_tensor = generate_anchors(x, model.detect.stride.flip((0,)), model.detect.grid_cell_size,
                                                        model.detect.grid_cell_offset, device='cpu', is_eval=True, mode='af')
        layers.get_anchors(anchor_points.reshape([-1]), stride_tensor)

    else:
        raise SystemExit('Model not supported')

os.system('echo "%d" | cat - %s > temp && mv temp %s' % (layers.wc, wts_file, wts_file))
