
import struct
import sys
from models import *
from models.models import *
from utils.utils import *
import argparse


def convert_cfg(fin, fo, classes, width, height):

    # Load the input file to a list
    with open(fin, 'r') as fin_obj:
        fin_lines = fin_obj.readlines()

    # Compute the number of filters and where to place them
    filters = int((int(classes) + 5) * 3)
    yolo_lines = [i for i,line in enumerate(fin_lines) if '[yolo]' in line]
    filter_lines = [i for i,line in enumerate(fin_lines) if 'filters' in line]
    lines_filter_before_yolo = []
    for i in yolo_lines:
        it = filter(lambda number: number < i, filter_lines)
        filtered_numbers = list(it)
        lines_filter_before_yolo.append(filtered_numbers[-1])

    # Write to output file with AlexeyAB described changes
    with open(fo, 'w') as fo_obj:
        for idx,line in enumerate(fin_lines):

            if 'width=' in line:
                fo_obj.write(f'width={width}'+'\n')

            elif 'height=' in line:
                fo_obj.write(f'height={height}'+'\n')

            elif 'classes=' in line:
                fo_obj.write(f'classes={classes}'+'\n')

            elif idx in lines_filter_before_yolo:
                fo_obj.write(f'filters={filters}'+'\n')

            else:
                fo_obj.write(line)


def classes_from_wts(weights):
    tmp_model = torch.load(weights, map_location='cpu')
    filters = [i for i in tmp_model['model'].values()][-1].shape
    classes = filters[0] / 3 - 5
    return int(classes)


def cfg_from_wts(weights):
    return 'cfg/' + weights.split('_')[0] + '.cfg'


def imsize_from_wts(weights):
    tmp_model = torch.load(weights, map_location='cpu')
    row = tmp_model['training_results'].split('\n')[0]
    print(row)
    imsize = int(row.split()[7])
    return (imsize, imsize)


def imsize_unpack(imsize):
    if len(imsize) == 1:
        width = height = imsize[0]
    elif len(imsize) == 2:
        width, height = imsize
    else:
        raise Exception('Too many size arguments: `-s width [height]`')
    assert height % 32 == 0
    assert width % 32 == 0
    return width, height


def layer_report(k_now, k_old, wt_so_far, v):
    """ This overengineered contraption prints out layer details through
    processing. Call before the weights are actually added to the .wts. For
    completion, call after loop has finished passing k_old='final'
    """
    if k_old == 'final':
        __, id, nname, __ = k_now.split('.')
        id = '(' + str(id) + ')'
        print('{0:<5s} {1:<15s}{2:<10d}{3:<25s}'.format(
            id, nname, wt_so_far, str(list(v.shape))))
        return

    __, nid, nname, __ = k_now.split('.')
    __, oid, __, __ = k_old.split('.')
    nid, oid = int(nid), int(oid)

    if nid - oid >= 1:
        id = '(' + str(oid) + ')'
        print('{0:<5s} {1:<15s}{2:<10d}{3:<25s}'.format(
            id, nname, wt_so_far, str(list(v.shape))))

    if nid - oid >= 2:
        for i in range(oid + 1, nid):
            id = '(' + str(i) + ')'
            print(id, '-')


if __name__ == '__main__':
    output_cfg = 'nvds-yolov4-tiny.cfg'
    output_wts = 'nvds-yolov4-tiny.wts'

    parser = argparse.ArgumentParser(description='''Create YOLOv4-tiny wts+cfg from pt+cfg''')
    parser.add_argument('--weights',   '-w',  type=str, required=True, help='Input .pt weights')
    parser.add_argument('--imsize',    '-s',  type=int, nargs='+', help='Network size: width [height]')
    parser.add_argument('--nclasses',  '-n',  type=int, help='Detect how many classes')
    parser.add_argument('--inputcfg',  '-c',  type=str, help='Path to input .cfg, defaults to /cfg dir')

    # Args to dict and purge Nones to use .get(val, default) later
    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}
    weights   = args.get('weights')
    imsize    = args.get('imsize',  imsize_from_wts(weights))
    input_cfg = args.get('inputcfg', cfg_from_wts(weights))
    classes   = args.get('nclasses', classes_from_wts(weights))

    # Fix cfg with AlexeyAB configuration
    width, height = imsize_unpack(imsize)
    convert_cfg(input_cfg, output_cfg, classes, width, height)
    model = Darknet(output_cfg, (width, height))

    # Weight convert
    if weights.endswith('.pt'):
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:
        load_darknet_weights(model, weights)

    with open(output_wts, 'w') as f:
        f.write('{}\n'.format(len([k for k in model.state_dict().keys() if 'batch' not in k])))
        wt_so_far =0
        k_old = '0.0.0.0'
        for k, v in model.state_dict().items():
            layer_report(k, k_old, wt_so_far, v)
            if 'num_batches_tracked' not in k:
                vr = v.reshape(-1).numpy()
                f.write('{} {}'.format(k, len(vr)))
                for vv in vr:
                    f.write(' ')
                    f.write(struct.pack('>f',float(vv)).hex())
                    wt_so_far +=1
                f.write('\n')
            k_old = k
        layer_report(k, 'final', wt_so_far, v)

    print('\nFinished for config:')
    print('> weights:', weights)
    print('> imsize:', imsize)
    print('> classes:', classes)
    print('> input_cfg:', input_cfg)

