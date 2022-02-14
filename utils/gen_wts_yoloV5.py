import argparse
import yaml
import math
import os
import struct
import torch
from utils.torch_utils import select_device


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch YOLOv5 conversion")
    parser.add_argument("-w", "--weights", required=True, help="Input weights (.pt) file path (required)")
    parser.add_argument("-c", "--yaml", help="Input cfg (.yaml) file path")
    parser.add_argument("-mw", "--width", help="Model width (default = 640 / 1280 [P6])")
    parser.add_argument("-mh", "--height", help="Model height (default = 640 / 1280 [P6])")
    parser.add_argument("-mc", "--channels", help="Model channels (default = 3)")
    parser.add_argument("--p6", action="store_true", help="P6 model")
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit("Invalid weights file")
    if not args.yaml:
        args.yaml = ""
    if not args.width:
        args.width = 1280 if args.p6 else 640
    if not args.height:
        args.height = 1280 if args.p6 else 640
    if not args.channels:
        args.channels = 3
    return args.weights, args.yaml, args.width, args.height, args.channels, args.p6


def get_width(x, gw, divisor=8):
    return int(math.ceil((x * gw) / divisor)) * divisor


def get_depth(x, gd):
    if x == 1:
        return 1
    r = int(round(x * gd))
    if x * gd - int(x * gd) == 0.5 and int(x * gd) % 2 == 0:
        r -= 1
    return max(r, 1)


pt_file, yaml_file, model_width, model_height, model_channels, p6 = parse_args()

model_name = pt_file.split(".pt")[0]
wts_file = model_name + ".wts"
cfg_file = model_name + ".cfg"

if yaml_file == "":
    yaml_file = "models/" + model_name + ".yaml"
    if not os.path.isfile(yaml_file):
        yaml_file = "models/hub/" + model_name + ".yaml"
        if not os.path.isfile(yaml_file):
            raise SystemExit("YAML file not found")
elif not os.path.isfile(yaml_file):
    raise SystemExit("Invalid YAML file")

device = select_device("cpu")
model = torch.load(pt_file, map_location=device)["model"].float()

anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]
delattr(model.model[-1], "anchor_grid")
model.model[-1].register_buffer("anchor_grid", anchor_grid)

model.to(device).eval()

anchors = ""
masks = []

with open(wts_file, "w") as f:
    wts_write = ""
    conv_count = 0
    cv1 = ""
    cv3 = ""
    cv3_idx = 0
    sppf_idx = 11 if p6 else 9
    for k, v in model.state_dict().items():
        if not "num_batches_tracked" in k and not "anchors" in k and not "anchor_grid" in k:
            vr = v.reshape(-1).cpu().numpy()
            idx = int(k.split(".")[1])
            if ".cv1." in k and not ".m." in k and idx != sppf_idx:
                cv1 += "{} {} ".format(k, len(vr))
                for vv in vr:
                    cv1 += " "
                    cv1 += struct.pack(">f", float(vv)).hex()
                cv1 += "\n"
                conv_count += 1
            elif cv1 != "" and ".m." in k:
                wts_write += cv1
                cv1 = ""
            if ".cv3." in k:
                cv3 += "{} {} ".format(k, len(vr))
                for vv in vr:
                    cv3 += " "
                    cv3 += struct.pack(">f", float(vv)).hex()
                cv3 += "\n"
                cv3_idx = idx
                conv_count += 1
            elif cv3 != "" and cv3_idx != idx:
                wts_write += cv3
                cv3 = ""
                cv3_idx = 0
            if not ".cv3." in k and not (".cv1." in k and not ".m." in k and idx != sppf_idx):
                wts_write += "{} {} ".format(k, len(vr))
                for vv in vr:
                    wts_write += " "
                    wts_write += struct.pack(">f", float(vv)).hex()
                wts_write += "\n"
                conv_count += 1
        elif "anchor_grid" in k:
            vr = v.cpu().numpy().tolist()
            a = v.reshape(-1).cpu().numpy().astype(int).tolist()
            anchors = str(a)[1:-1]
            num = 0
            for m in vr:
                mask = []
                for _ in range(len(m)):
                    mask.append(num)
                    num += 1
                masks.append(mask)
    f.write("{}\n".format(conv_count))
    f.write(wts_write)

with open(cfg_file, "w") as c:
    with open(yaml_file, "r") as f:
        nc = 0
        depth_multiple = 0
        width_multiple = 0
        detections = []
        layers = []
        f = yaml.load(f,Loader=yaml.FullLoader)
        c.write("[net]\n")
        c.write("width=%d\n" % model_width)
        c.write("height=%d\n" % model_height)
        c.write("channels=%d\n" % model_channels)
        for l in f:
            if l == "nc":
                nc = f[l]
            elif l == "depth_multiple":
                depth_multiple = f[l]
            elif l == "width_multiple":
                width_multiple = f[l]
            elif l == "backbone" or l == "head":
                for v in f[l]:
                    if v[2] == "Conv":
                        layer = ""
                        blocks = 0
                        layer += "\n[convolutional]\n"
                        layer += "batch_normalize=1\n"
                        layer += "filters=%d\n" % get_width(v[3][0], width_multiple)
                        layer += "size=%d\n" % v[3][1]
                        layer += "stride=%d\n" % v[3][2]
                        layer += "pad=1\n"
                        layer += "activation=silu\n"
                        blocks += 1
                        layers.append([layer, blocks])
                    elif v[2] == "C3":
                        layer = ""
                        blocks = 0
                        layer += "\n# C3\n"
                        # SPLIT
                        layer += "\n[convolutional]\n"
                        layer += "batch_normalize=1\n"
                        layer += "filters=%d\n" % get_width(v[3][0] / 2, width_multiple)
                        layer += "size=1\n"
                        layer += "stride=1\n"
                        layer += "pad=1\n"
                        layer += "activation=silu\n"
                        blocks += 1
                        layer += "\n[route]\n"
                        layer += "layers=-2\n"
                        blocks += 1
                        layer += "\n[convolutional]\n"
                        layer += "batch_normalize=1\n"
                        layer += "filters=%d\n" % get_width(v[3][0] / 2, width_multiple)
                        layer += "size=1\n"
                        layer += "stride=1\n"
                        layer += "pad=1\n"
                        layer += "activation=silu\n"
                        blocks += 1
                        # Residual Block
                        if len(v[3]) == 1 or v[3][1] == True:
                            for _ in range(get_depth(v[1], depth_multiple)):
                                layer += "\n[convolutional]\n"
                                layer += "batch_normalize=1\n"
                                layer += "filters=%d\n" % get_width(v[3][0] / 2, width_multiple)
                                layer += "size=1\n"
                                layer += "stride=1\n"
                                layer += "pad=1\n"
                                layer += "activation=silu\n"
                                blocks += 1
                                layer += "\n[convolutional]\n"
                                layer += "batch_normalize=1\n"
                                layer += "filters=%d\n" % get_width(v[3][0] / 2, width_multiple)
                                layer += "size=3\n"
                                layer += "stride=1\n"
                                layer += "pad=1\n"
                                layer += "activation=silu\n"
                                blocks += 1
                                layer += "\n[shortcut]\n"
                                layer += "from=-3\n"
                                layer += "activation=linear\n"
                                blocks += 1
                            # Merge
                            layer += "\n[route]\n"
                            layer += "layers=-1, -%d\n" % (3 * get_depth(v[1], depth_multiple) + 3)
                            blocks += 1
                        else:
                            for _ in range(get_depth(v[1], depth_multiple)):
                                layer += "\n[convolutional]\n"
                                layer += "batch_normalize=1\n"
                                layer += "filters=%d\n" % get_width(v[3][0] / 2, width_multiple)
                                layer += "size=1\n"
                                layer += "stride=1\n"
                                layer += "pad=1\n"
                                layer += "activation=silu\n"
                                blocks += 1
                                layer += "\n[convolutional]\n"
                                layer += "batch_normalize=1\n"
                                layer += "filters=%d\n" % get_width(v[3][0] / 2, width_multiple)
                                layer += "size=3\n"
                                layer += "stride=1\n"
                                layer += "pad=1\n"
                                layer += "activation=silu\n"
                                blocks += 1
                            # Merge
                            layer += "\n[route]\n"
                            layer += "layers=-1, -%d\n" % (2 * get_depth(v[1], depth_multiple) + 3)
                            blocks += 1
                        # Transition
                        layer += "\n[convolutional]\n"
                        layer += "batch_normalize=1\n"
                        layer += "filters=%d\n" % get_width(v[3][0], width_multiple)
                        layer += "size=1\n"
                        layer += "stride=1\n"
                        layer += "pad=1\n"
                        layer += "activation=silu\n"
                        layer += "\n##########\n"
                        blocks += 1
                        layers.append([layer, blocks])
                    elif v[2] == "SPPF":
                        layer = ""
                        blocks = 0
                        layer += "\n# SPPF\n"
                        layer += "\n[convolutional]\n"
                        layer += "batch_normalize=1\n"
                        layer += "filters=%d\n" % (get_width(v[3][0], width_multiple) / 2)
                        layer += "size=1\n"
                        layer += "stride=1\n"
                        layer += "pad=1\n"
                        layer += "activation=silu\n"
                        blocks += 1
                        layer += "\n[maxpool]\n"
                        layer += "stride=1\n"
                        layer += "size=%d\n" % v[3][1]
                        blocks += 1
                        layer += "\n[maxpool]\n"
                        layer += "stride=1\n"
                        layer += "size=%d\n" % v[3][1]
                        blocks += 1
                        layer += "\n[maxpool]\n"
                        layer += "stride=1\n"
                        layer += "size=%d\n" % v[3][1]
                        blocks += 1
                        layer += "\n[route]\n"
                        layer += "layers=-4, -3, -2, -1\n"
                        blocks += 1
                        layer += "\n[convolutional]\n"
                        layer += "batch_normalize=1\n"
                        layer += "filters=%d\n" % get_width(v[3][0], width_multiple)
                        layer += "size=1\n"
                        layer += "stride=1\n"
                        layer += "pad=1\n"
                        layer += "activation=silu\n"
                        layer += "\n##########\n"
                        blocks += 1
                        layers.append([layer, blocks])
                    elif v[2] == "nn.Upsample":
                        layer = ""
                        blocks = 0
                        layer += "\n[upsample]\n"
                        layer += "stride=%d\n" % v[3][1]
                        blocks += 1
                        layers.append([layer, blocks])
                    elif v[2] == "Concat":
                        layer = ""
                        blocks = 0
                        route = v[0][1]
                        r = 0
                        if route > 0:
                            for i, item in enumerate(layers):
                                if i <= route:
                                    r += item[1]
                                else:
                                    break
                        else:
                            route = len(layers) + route
                            for i, item in enumerate(layers):
                                if i <= route:
                                    r += item[1]
                                else:
                                    break
                        layer += "\n# Concat\n"
                        layer += "\n[route]\n"
                        layer += "layers=-1, %d\n" % (r - 1)
                        layer += "\n##########\n"
                        blocks += 1
                        layers.append([layer, blocks])
                    elif v[2] == "Detect":
                        for i, n in enumerate(v[0]):
                            layer = ""
                            blocks = 0
                            r = 0
                            for j, item in enumerate(layers):
                                if j <= n:
                                    r += item[1]
                                else:
                                    break
                            layer += "\n# Detect\n"
                            layer += "\n[route]\n"
                            layer += "layers=%d\n" % (r - 1)
                            blocks += 1
                            layer += "\n[convolutional]\n"
                            layer += "size=1\n"
                            layer += "stride=1\n"
                            layer += "pad=1\n"
                            layer += "filters=%d\n" % ((nc + 5) * 3)
                            layer += "activation=logistic\n"
                            blocks += 1
                            layer += "\n[yolo]\n"
                            layer += "mask=%s\n" % str(masks[i])[1:-1]
                            layer += "anchors=%s\n" % anchors
                            layer += "classes=%d\n" % nc
                            layer += "num=%d\n" % num
                            layer += "scale_x_y=2.0\n"
                            layer += "beta_nms=0.6\n"
                            layer += "new_coords=1\n"
                            layer += "\n##########\n"
                            blocks += 1
                            layers.append([layer, blocks])
        for layer in layers:
            c.write(layer[0])
