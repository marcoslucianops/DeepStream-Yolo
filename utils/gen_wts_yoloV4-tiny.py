
import struct
import sys
from models import *
from models.models import *
from utils.utils import *
import argparse


class CfgFixer:

    def __init__(self, inputcfg, classes, width, height=None):

        # Define dump for the input file + class retrieval point
        # file_out = inputcfg.split('.')[0] + '_adjusted.cfg'
        file_out = inputcfg.split('/')[1]
        self.file_out = file_out

        # Receive input width and stuff
        width = int(width)
        height = int(height) if height else width
        assert height % 32 == 0
        assert width % 32 == 0

        # Load the input file to a list
        file_content = []
        with open(inputcfg, 'r') as file_object:
            for line in file_object:
                file_content.append(line)

        # Compute the number of filters and where to place them
        filters = int((int(classes)+5)*3)
        lines_filter_before_yolo = self.exact_filter_line(file_content)

        # Actually modify the file with AlexeyAB described changes
        with open(file_out, 'w') as file_object:
            for idx,line in enumerate(file_content):

                if any(['batch=' in line and 'batch_' not in line and '_batch' not in line,
                       'subdivisions=' in line,
                       'max_batche=s' in line,
                       'steps=' in line and 'policy' not in line
                        ]):
                    # We don't do that here
                    pass

                elif 'width=' in line:
                    file_object.write(f'width={width}'+'\n')

                elif 'height=' in line:
                    file_object.write(f'height={height}'+'\n')

                elif 'classes=' in line:
                    file_object.write(f'classes={classes}'+'\n')

                elif idx in lines_filter_before_yolo:
                    file_object.write(f'filters={filters}'+'\n')

                else:
                    file_object.write(line)



    @staticmethod
    def exact_filter_line(file_content):
        """ Get the exact line of the last filter line before each yolo layer """

        # Scan the file for the [yolo] and filter= lines
        yolo_lines = [i for i,line in enumerate(file_content) if '[yolo]' in line]
        filter_lines = [i for i,line in enumerate(file_content) if 'filters' in line]

        # Get only the filter lines prior to yolo layers
        exact_filter_list = []
        for i in yolo_lines:
            it = filter(lambda number: number < i, filter_lines)
            filtered_numbers = list(it)
            exact_filter_list.append(filtered_numbers[-1])

        return exact_filter_list


def route_fix(file_out):
    route_fix_context = False

    # Load file content into memory
    with open(file_out, 'r') as file_object:
        file_content = []
        for line in file_object:
            file_content.append(line)

    # Truncate and rewrite file, translating [route_lhalf] -> [route]
    with open(file_out, 'w') as file_object:
        for idx,line in enumerate(file_content):
            if '[route_lhalf]' in line:
                route_fix_context = True
                file_object.write('[route]' + '\n')

            elif route_fix_context:
                if '[' in line:
                    route_fix_context = False
                    file_object.write(line)

                elif 'layers' in line:
                    file_object.write(line)
                    file_object.write('groups=2'+'\n')
                    file_object.write('group_id=1'+'\n'*2)

            else:
                file_object.write(line)


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
    imsize = row.split()[7]
    return imsize


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Create YOLOv4-tiny wts+cfg''')
    parser.add_argument('--weights',  '-w',  type=str, required=True, help='network size')
    parser.add_argument('--nclasses', '-n',  type=int, help='Detect how many classes')
    parser.add_argument('--imsize',   '-s',  type=int, help='network size')
    parser.add_argument('--inputcfg', '-c',  type=str, help='Path to default .cfg')

    # Args to dict and purge Nones to use .get(val, default) later
    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}

    weights  = args.get('weights')
    imsize   = args.get('imsize',   imsize_from_wts(weights))
    inputcfg = args.get('inputcfg', cfg_from_wts(weights))
    classes  = args.get('nclasses', classes_from_wts(weights))

    print(weights, imsize, inputcfg, classes)


    # Fix cdf with AlexeyAB configuration
    newcfg = CfgFixer(inputcfg, classes, imsize).file_out
    model = Darknet(newcfg, (imsize, imsize))
    route_fix(newcfg)

    dev = 'cpu'
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=dev)['model'])
    else:
        raise Exception('Unknown weight type / file not found')
        # load_darknet_weights(model, weights)

    f = open('yolov4-tiny.wts', 'w')
    f.write('{}\n'.format(len(model.state_dict().keys())))
    wt_so_far =0
    k_old = '0.0'

    index = lambda k, i=1: k.split('.')[i]
    iindex = lambda k, i=1: int(k.split('.')[i])
    diff = lambda x, y: int(index(x)) - int(index(y))

    for k, v in model.state_dict().items():

        # Print summary of PREVIOUS layer
        if diff(k, k_old) >= 1:
            print(index(k_old), wt_so_far)

        # Fill in intermediate layer indices
        if diff(k, k_old) >= 2:
            for i in range(iindex(k_old)+1, iindex(k)):
                print(i, '-')

        if 'num_batches_tracked' not in k:
            # print('\t', v.shape)
            vr = v.reshape(-1).numpy()
            f.write('{} {}'.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f',float(vv)).hex())
                wt_so_far +=1
            f.write('\n')

        # print(index(k), k, wt_so_far)
        k_old = k

    # Last layer manually printed out
    print(index(k_old), wt_so_far)
    print(iindex(k_old) +1 , '-')



