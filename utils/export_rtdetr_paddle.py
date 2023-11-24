import os
import sys
import warnings
import onnx
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import load_config, merge_config
from ppdet.utils.check import check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.engine import Trainer


class DeepStreamOutput(nn.Layer):
    def __init__(self, img_size, use_focal_loss):
        self.img_size = img_size
        self.use_focal_loss = use_focal_loss
        super().__init__()

    def forward(self, x):
        boxes = x['bbox']
        out_shape = paddle.to_tensor([[*self.img_size]]).flip(1).tile([1, 2]).unsqueeze(1)
        boxes *= out_shape
        bbox_num = F.sigmoid(x['bbox_num']) if self.use_focal_loss else F.softmax(x['bbox_num'])[:, :, :-1]
        scores = paddle.max(bbox_num, 2, keepdim=True)
        classes = paddle.cast(paddle.argmax(bbox_num, 2, keepdim=True), dtype='float32')
        return boxes, scores, classes
    

def suppress_warnings():
    warnings.filterwarnings('ignore')


def rtdetr_paddle_export(FLAGS):
    cfg = load_config(FLAGS.config)
    FLAGS.opt['weights'] = FLAGS.weights
    FLAGS.opt['exclude_nms'] = True
    FLAGS.opt['exclude_post_process'] = True
    merge_config(FLAGS.opt)
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
    return trainer.cfg, static_model


def main(FLAGS):
    suppress_warnings()

    print('\nStarting: %s' % FLAGS.weights)

    print('\nOpening RT-DETR Paddle model\n')

    paddle.set_device('cpu')
    cfg, model = rtdetr_paddle_export(FLAGS)

    img_size = [cfg.eval_size[1], cfg.eval_size[0]]

    model = nn.Sequential(model, DeepStreamOutput(img_size, cfg.use_focal_loss))

    onnx_input_im = {}
    onnx_input_im['image'] = paddle.static.InputSpec(shape=[FLAGS.batch, 3, *img_size], dtype='float32', name='image')
    onnx_output_file = cfg.filename + '.onnx'

    print('\nExporting the model to ONNX\n')
    paddle.onnx.export(model, cfg.filename, input_spec=[onnx_input_im], opset_version=FLAGS.opset)

    if FLAGS.simplify:
        print('\nSimplifying the ONNX model')
        import onnxsim
        model_onnx = onnx.load(onnx_output_file)
        model_onnx, _ = onnxsim.simplify(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print('\nDone: %s\n' % onnx_output_file)


def parse_args():
    parser = ArgsParser()
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pdparams) file path (required)')
    parser.add_argument('--slim_config', default=None, type=str, help='Slim configuration file of slim method')
    parser.add_argument('--opset', type=int, default=16, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Static batch-size')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('\nInvalid weights file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('\nCannot set dynamic batch-size and static batch-size at same time')
    elif args.dynamic:
        args.batch = None
    return args


if __name__ == '__main__':
    FLAGS = parse_args()
    sys.exit(main(FLAGS))
