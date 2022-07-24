import argparse
import os
import struct
import torch
from utils.torch_utils import select_device
from models.models import Darknet


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch YOLOR conversion (main branch)')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-c', '--cfg', default='', help='Input cfg (.cfg) file path')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if args.cfg != '' and not os.path.isfile(args.cfg):
        raise SystemExit('Invalid cfg file')
    return args.weights, args.cfg


pt_file, cfg_file = parse_args()


model_name = os.path.basename(pt_file).split('.pt')[0]
wts_file = model_name + '.wts' if 'yolor' in model_name else 'yolor_' + model_name + '.wts'
new_cfg_file = model_name + '.cfg' if 'yolor' in model_name else 'yolor_' + model_name + '.cfg'

if cfg_file == '':
    cfg_file = 'cfg/' + model_name + '.cfg'
    if not os.path.isfile(cfg_file):
        raise SystemExit('CFG file not found')
elif not os.path.isfile(cfg_file):
    raise SystemExit('Invalid CFG file')

device = select_device('cpu')
model = Darknet(cfg_file).to(device)
model.load_state_dict(torch.load(pt_file, map_location=device)['model'])
model.to(device).eval()

with open(wts_file, 'w') as f:
    wts_write = ''
    conv_count = 0
    for k, v in model.state_dict().items():
        if 'num_batches_tracked' not in k:
            vr = v.reshape(-1).cpu().numpy()
            wts_write += '{} {} '.format(k, len(vr))
            for vv in vr:
                wts_write += ' '
                wts_write += struct.pack('>f', float(vv)).hex()
            wts_write += '\n'
            conv_count += 1
    f.write('{}\n'.format(conv_count))
    f.write(wts_write)

if not os.path.isfile(new_cfg_file):
    os.system('cp %s %s' % (cfg_file, new_cfg_file))
