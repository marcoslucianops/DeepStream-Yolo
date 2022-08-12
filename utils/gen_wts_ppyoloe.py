import os
import struct
import paddle
import numpy as np
from ppdet.core.workspace import load_config, merge_config
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.engine import Trainer
from ppdet.slim import build_slim_model

class Layers(object):
    def __init__(self, size, fw, fc, letter_box):
        self.blocks = [0 for _ in range(300)]
        self.current = -1

        self.backbone_outs = []
        self.neck_fpn_feats = []
        self.neck_pan_feats = []
        self.yolo_head_cls = []
        self.yolo_head_reg = []

        self.width = size[0] if len(size) == 1 else size[1]
        self.height = size[0]
        self.letter_box = letter_box

        self.fw = fw
        self.fc = fc
        self.wc = 0

        self.net()

    def ConvBNLayer(self, child):
        self.current += 1

        self.convolutional(child, act='swish')

    def CSPResStage(self, child, ret):
        self.current += 1

        if child.conv_down is not None:
            self.convolutional(child.conv_down, act='swish')
        self.convolutional(child.conv1, act='swish')
        self.route('-2')
        self.convolutional(child.conv2, act='swish')
        idx = -3
        for m in child.blocks:
            self.convolutional(m.conv1, act='swish')
            self.convolutional(m.conv2, act='swish')
            self.shortcut(-3)
            idx -= 3
        self.route('%d, -1' % idx)
        if child.attn is not None:
            self.reduce((1, 2), mode='mean', keepdim=True)
            self.convolutional(child.attn.fc, act='hardsigmoid')
            self.shortcut(-3, ew='mul')
        self.convolutional(child.conv3, act='swish')
        if ret is True:
            self.backbone_outs.append(self.current)

    def CSPStage(self, child, stage):
        self.current += 1

        self.convolutional(child.conv1, act='swish')
        self.route('-2')
        self.convolutional(child.conv2, act='swish')
        idx = -3
        for m in child.convs:
            if m.__class__.__name__ == 'BasicBlock':
                self.convolutional(m.conv1, act='swish')
                self.convolutional(m.conv2, act='swish')
                idx -= 2
            elif m.__class__.__name__ == 'SPP':
                self.maxpool(m.pool0)
                self.route('-2')
                self.maxpool(m.pool1)
                self.route('-4')
                self.maxpool(m.pool2)
                self.route('-6, -5, -3, -1')
                self.convolutional(m.conv, act='swish')
                idx -= 7
        self.route('%d, -1' % idx)
        self.convolutional(child.conv3, act='swish')
        if stage == 'fpn':
            self.neck_fpn_feats.append(self.current)
        elif stage == 'pan':
            self.neck_pan_feats.append(self.current)

    def Concat(self, route):
        self.current += 1

        r = self.get_route(route)
        self.route('-1, %d' % r)

    def Upsample(self):
        self.current += 1

        self.upsample()

    def AvgPool2d(self, route=None):
        self.current += 1

        if route is not None:
            r = self.get_route(route)
            self.route('%d' % r)
        self.avgpool()

    def ESEAttn(self, child, route=0):
        self.current += 1

        if route < 0:
            self.route('%d' % route)
        self.convolutional(child.fc, act='sigmoid')
        self.shortcut(route - 3, ew='mul')
        self.convolutional(child.conv, act='swish')
        if route == 0:
            self.shortcut(-5)

    def Conv2D(self, child, act='linear'):
        self.current += 1

        self.convolutional(child, act=act)

    def Shuffle(self, reshape=None, transpose1=None, transpose2=None, route=None, output=''):
        self.current += 1

        r = 0
        if route is not None:
            r = self.get_route(route)
        self.shuffle(reshape=reshape, transpose1=transpose1, transpose2=transpose2, route=r)
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
        lb = 'letter_box=1\n' if self.letter_box else ''

        self.fc.write('[net]\n' +
                      'width=%d\n' % self.width +
                      'height=%d\n' % self.height +
                      'channels=3\n' +
                      lb)

    def convolutional(self, cv, act='linear', detect=False):
        self.blocks[self.current] += 1

        self.get_state_dict(cv.state_dict())

        if cv.__class__.__name__ == 'Conv2D':
            filters = cv._out_channels
            size = cv._kernel_size
            stride = cv._stride
            pad = cv._padding
            groups = cv._groups
            bias = cv.bias
            bn = False
        else:
            filters = cv.conv._out_channels
            size = cv.conv._kernel_size
            stride = cv.conv._stride
            pad = cv.conv._padding
            groups = cv.conv._groups
            bias = cv.conv.bias
            bn = True if hasattr(cv, 'bn') else False

        if detect:
            act = 'logistic'

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

    def reduce(self, dim, mode='mean', keepdim=False):
        self.blocks[self.current] += 1

        self.fc.write('\n[reduce]\n' +
                      'mode=%s\n' % mode +
                      'axes=%s\n' % str(dim)[1:-1] +
                      'keep=%d\n' % keepdim)

    def maxpool(self, m):
        self.blocks[self.current] += 1

        stride = m.stride
        size = m.ksize
        mode = m.ceil_mode

        m = 'maxpool_up' if mode else 'maxpool'

        self.fc.write('\n[%s]\n' % m +
                      'stride=%d\n' % stride +
                      'size=%d\n' % size)

    def upsample(self):
        self.blocks[self.current] += 1

        stride = 2

        self.fc.write('\n[upsample]\n' +
                      'stride=%d\n' % stride)

    def avgpool(self):
        self.blocks[self.current] += 1

        self.fc.write('\n[avgpool]\n')

    def shuffle(self, reshape=None, transpose1=None, transpose2=None, route=None):
        self.blocks[self.current] += 1

        r = 'reshape=%s\n' % str(reshape)[1:-1] if reshape is not None else ''
        t1 = 'transpose1=%s\n' % str(transpose1)[1:-1] if transpose1 is not None else ''
        t2 = 'transpose2=%s\n' % str(transpose2)[1:-1] if transpose2 is not None else ''
        f = 'from=%d\n' % route if route is not None else ''

        self.fc.write('\n[shuffle]\n' +
                      r +
                      t1 +
                      t2 +
                      f)

    def softmax(self, axes):
        self.blocks[self.current] += 1

        self.fc.write('\n[softmax]\n' +
                      'axes=%d\n' % axes)

    def yolo(self, output):
        self.blocks[self.current] += 1

        self.fc.write('\n[%s]\n' % output)

    def get_state_dict(self, state_dict):
        for k, v in state_dict.items():
            vr = v.reshape([-1]).numpy()
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


def export_model():
    paddle.set_device('cpu')

    FLAGS = parse_args()

    cfg = load_config(FLAGS.config)

    FLAGS.opt['weights'] = FLAGS.weights
    FLAGS.opt['exclude_nms'] = True

    if 'norm_type' in cfg and cfg['norm_type'] == 'sync_bn':
        FLAGS.opt['norm_type'] = 'bn'
    merge_config(FLAGS.opt)

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')

    merge_config(FLAGS.opt)
    check_config(cfg)
    check_version()

    trainer = Trainer(cfg, mode='test')
    trainer.load_weights(cfg.weights)

    trainer.model.eval()

    if not os.path.exists('.tmp'):
        os.makedirs('.tmp')

    static_model, _ = trainer._get_infer_cfg_and_input_spec('.tmp')

    os.system('rm -r .tmp')

    return cfg, static_model


def parse_args():
    parser = ArgsParser()
    parser.add_argument('-w', '--weights', required=True, type=str, help='Input weights (.pdparams) file path (required)')
    parser.add_argument('--slim_config', default=None, type=str, help='Slim configuration file of slim method')
    args = parser.parse_args()
    return args


cfg, model = export_model()

model_name = cfg.filename
inference_size = (cfg.eval_height, cfg.eval_width)
letter_box = False

for sample_transforms in cfg['EvalReader']['sample_transforms']:
    if 'Resize' in sample_transforms:
        letter_box = sample_transforms['Resize']['keep_ratio']

backbone = cfg[cfg.architecture]['backbone']
neck = cfg[cfg.architecture]['neck']
yolo_head = cfg[cfg.architecture]['yolo_head']

wts_file = model_name + '.wts' if 'ppyoloe' in model_name else 'ppyoloe_' + model_name + '.wts'
cfg_file = model_name + '.cfg' if 'ppyoloe' in model_name else 'ppyoloe_' + model_name + '.cfg'

with open(wts_file, 'w') as fw, open(cfg_file, 'w') as fc:
    layers = Layers(inference_size, fw, fc, letter_box)

    if backbone == 'CSPResNet':
        layers.fc.write('\n# CSPResNet\n')

        for child in model.backbone.stem:
            layers.ConvBNLayer(child)
        for i, child in enumerate(model.backbone.stages):
            ret = True if i in model.backbone.return_idx else False
            layers.CSPResStage(child, ret)
    else:
        raise SystemExit('Model not supported')

    if neck == 'CustomCSPPAN':
        layers.fc.write('\n# CustomCSPPAN\n')

        blocks = layers.backbone_outs[::-1]
        for i, block in enumerate(blocks):
            if i > 0:
                layers.Concat(block)
            layers.CSPStage(model.neck.fpn_stages[i][0], 'fpn')
            if i < model.neck.num_blocks - 1:
                layers.ConvBNLayer(model.neck.fpn_routes[i])
                layers.Upsample()
        layers.neck_pan_feats = [layers.neck_fpn_feats[-1], ]
        for i in reversed(range(model.neck.num_blocks - 1)):
            layers.ConvBNLayer(model.neck.pan_routes[i])
            layers.Concat(layers.neck_fpn_feats[i])
            layers.CSPStage(model.neck.pan_stages[i][0], 'pan')
        layers.neck_pan_feats = layers.neck_pan_feats[::-1]
    else:
        raise SystemExit('Model not supported')

    if yolo_head == 'PPYOLOEHead':
        layers.fc.write('\n# PPYOLOEHead\n')

        for i, feat in enumerate(layers.neck_pan_feats):
            if i > 0:
                layers.AvgPool2d(route=feat)
            else:
                layers.AvgPool2d()
            layers.ESEAttn(model.yolo_head.stem_cls[i])
            layers.Conv2D(model.yolo_head.pred_cls[i], act='sigmoid')
            layers.Shuffle(reshape=[model.yolo_head.num_classes, 0], route=feat, output='cls')
            layers.ESEAttn(model.yolo_head.stem_reg[i], route=-7)
            layers.Conv2D(model.yolo_head.pred_reg[i])
            layers.Shuffle(reshape=[4, model.yolo_head.reg_max + 1, 0], transpose2=[1, 0, 2], route=feat)
            layers.SoftMax(0)
            layers.Conv2D(model.yolo_head.proj_conv)
            layers.Shuffle(reshape=[4, 0], route=feat, output='reg')
        layers.Detect('cls')
        layers.Detect('reg')
        layers.get_anchors(model.yolo_head.anchor_points.reshape([-1]), model.yolo_head.stride_tensor)

    else:
        raise SystemExit('Model not supported')

os.system('echo "%d" | cat - %s > temp && mv temp %s' % (layers.wc, wts_file, wts_file))
