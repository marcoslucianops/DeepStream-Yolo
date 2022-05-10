import argparse
import yaml
import math
import os
import struct
import torch
from utils.torch_utils import select_device


class YoloLayers():
    def get_route(self, n, layers):
        route = 0
        for i, layer in enumerate(layers):
            if i <= n:
                route += layer[1]
            else:
                break
        return route

    def route(self, layers=""):
        return "\n[route]\n" + \
               "layers=%s\n" % layers

    def reorg(self):
        return "\n[reorg]\n"

    def shortcut(self, route=-1, activation="linear"):
        return "\n[shortcut]\n" + \
               "from=%d\n" % route + \
               "activation=%s\n" % activation

    def maxpool(self, stride=1, size=1):
        return "\n[maxpool]\n" + \
               "stride=%d\n" % stride + \
               "size=%d\n" % size

    def upsample(self, stride=1):
        return "\n[upsample]\n" + \
               "stride=%d\n" % stride

    def convolutional(self, bn=False, size=1, stride=1, pad=1, filters=1, groups=1, activation="linear"):
        b = "batch_normalize=1\n" if bn is True else ""
        g = "groups=%d\n" % groups if groups > 1 else ""
        return "\n[convolutional]\n" + \
               b + \
               "filters=%d\n" % filters + \
               "size=%d\n" % size + \
               "stride=%d\n" % stride + \
               "pad=%d\n" % pad + \
               g + \
               "activation=%s\n" % activation

    def yolo(self, mask="", anchors="", classes=80, num=3):
        return "\n[yolo]\n" + \
               "mask=%s\n" % mask + \
               "anchors=%s\n" % anchors + \
               "classes=%d\n" % classes + \
               "num=%d\n" % num + \
               "scale_x_y=2.0\n" + \
               "beta_nms=0.6\n" + \
               "new_coords=1\n"


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch YOLOv5 conversion")
    parser.add_argument("-w", "--weights", required=True, help="Input weights (.pt) file path (required)")
    parser.add_argument("-c", "--yaml", help="Input cfg (.yaml) file path")
    parser.add_argument("-mw", "--width", type=int, help="Model width (default = 640 / 1280 [P6])")
    parser.add_argument("-mh", "--height", type=int, help="Model height (default = 640 / 1280 [P6])")
    parser.add_argument("-mc", "--channels", type=int, help="Model channels (default = 3)")
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
wts_file = model_name + ".wts" if "yolov5" in model_name else "yolov5_" + model_name + ".wts"
cfg_file = model_name + ".cfg" if "yolov5" in model_name else "yolov5_" + model_name + ".cfg"

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

for k, v in model.state_dict().items():
    if "anchor_grid" in k:
        vr = v.cpu().numpy().tolist()
        a = v.reshape(-1).cpu().numpy().astype(float).tolist()
        anchors = str(a)[1:-1]
        num = 0
        for m in vr:
            mask = []
            for _ in range(len(m)):
                mask.append(num)
                num += 1
            masks.append(mask)

spp_idx = 0

with open(cfg_file, "w") as c:
    with open(yaml_file, "r", encoding="utf-8") as f:
        c.write("[net]\n")
        c.write("width=%d\n" % model_width)
        c.write("height=%d\n" % model_height)
        c.write("channels=%d\n" % model_channels)
        nc = 0
        depth_multiple = 0
        width_multiple = 0
        layers = []
        yoloLayers = YoloLayers()
        f = yaml.load(f, Loader=yaml.FullLoader)
        for topic in f:
            if topic == "nc":
                nc = f[topic]
            elif topic == "depth_multiple":
                depth_multiple = f[topic]
            elif topic == "width_multiple":
                width_multiple = f[topic]
            elif topic == "backbone" or topic == "head":
                for v in f[topic]:
                    if v[2] == "Focus":
                        layer = "\n# Focus\n"
                        blocks = 0
                        layer += yoloLayers.reorg()
                        blocks += 1
                        layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple), size=v[3][1],
                                                          activation="silu")
                        blocks += 1
                        layers.append([layer, blocks])
                    if v[2] == "Conv":
                        layer = "\n# Conv\n"
                        blocks = 0
                        layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple), size=v[3][1],
                                                          stride=v[3][2], activation="silu")
                        blocks += 1
                        layers.append([layer, blocks])
                    elif v[2] == "C3":
                        layer = "\n# C3\n"
                        blocks = 0
                        # SPLIT
                        layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple) / 2,
                                                          activation="silu")
                        blocks += 1
                        layer += yoloLayers.route(layers="-2")
                        blocks += 1
                        layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple) / 2,
                                                          activation="silu")
                        blocks += 1
                        # Residual Block
                        if len(v[3]) == 1 or v[3][1] is True:
                            for _ in range(get_depth(v[1], depth_multiple)):
                                layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple) / 2,
                                                                  activation="silu")
                                blocks += 1
                                layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple) / 2,
                                                                  size=3, activation="silu")
                                blocks += 1
                                layer += yoloLayers.shortcut(route=-3)
                                blocks += 1
                            # Merge
                            layer += yoloLayers.route(layers="-1, -%d" % (3 * get_depth(v[1], depth_multiple) + 3))
                            blocks += 1
                        else:
                            for _ in range(get_depth(v[1], depth_multiple)):
                                layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple) / 2,
                                                                  activation="silu")
                                blocks += 1
                                layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple) / 2,
                                                                  size=3, activation="silu")
                                blocks += 1
                            # Merge
                            layer += yoloLayers.route(layers="-1, -%d" % (2 * get_depth(v[1], depth_multiple) + 3))
                            blocks += 1
                        # Transition
                        layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple),
                                                          activation="silu")
                        blocks += 1
                        layers.append([layer, blocks])
                    elif v[2] == "SPP":
                        spp_idx = len(layers)
                        layer = "\n# SPP\n"
                        blocks = 0
                        layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple) / 2,
                                                          activation="silu")
                        blocks += 1
                        layer += yoloLayers.maxpool(size=v[3][1][0])
                        blocks += 1
                        layer += yoloLayers.route(layers="-2")
                        blocks += 1
                        layer += yoloLayers.maxpool(size=v[3][1][1])
                        blocks += 1
                        layer += yoloLayers.route(layers="-4")
                        blocks += 1
                        layer += yoloLayers.maxpool(size=v[3][1][2])
                        blocks += 1
                        layer += yoloLayers.route(layers="-6, -5, -3, -1")
                        blocks += 1
                        layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple),
                                                          activation="silu")
                        blocks += 1
                        layers.append([layer, blocks])
                    elif v[2] == "SPPF":
                        spp_idx = len(layers)
                        layer = "\n# SPPF\n"
                        blocks = 0
                        layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple) / 2,
                                                          activation="silu")
                        blocks += 1
                        layer += yoloLayers.maxpool(size=v[3][1])
                        blocks += 1
                        layer += yoloLayers.maxpool(size=v[3][1])
                        blocks += 1
                        layer += yoloLayers.maxpool(size=v[3][1])
                        blocks += 1
                        layer += yoloLayers.route(layers="-4, -3, -2, -1")
                        blocks += 1
                        layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple),
                                                          activation="silu")
                        blocks += 1
                        layers.append([layer, blocks])
                    elif v[2] == "nn.Upsample":
                        layer = "\n# nn.Upsample\n"
                        blocks = 0
                        layer += yoloLayers.upsample(stride=v[3][1])
                        blocks += 1
                        layers.append([layer, blocks])
                    elif v[2] == "Concat":
                        route = v[0][1]
                        route = yoloLayers.get_route(route, layers) if route > 0 else \
                            yoloLayers.get_route(len(layers) + route, layers)
                        layer = "\n# Concat\n"
                        blocks = 0
                        layer += yoloLayers.route(layers="-1, %d" % (route - 1))
                        blocks += 1
                        layers.append([layer, blocks])
                    elif v[2] == "Detect":
                        for i, n in enumerate(v[0]):
                            route = yoloLayers.get_route(n, layers)
                            layer = "\n# Detect\n"
                            blocks = 0
                            layer += yoloLayers.route(layers="%d" % (route - 1))
                            blocks += 1
                            layer += yoloLayers.convolutional(filters=((nc + 5) * len(masks[i])), activation="logistic")
                            blocks += 1
                            layer += yoloLayers.yolo(mask=str(masks[i])[1:-1], anchors=anchors, classes=nc, num=num)
                            blocks += 1
                            layers.append([layer, blocks])
        for layer in layers:
            c.write(layer[0])

with open(wts_file, "w") as f:
    wts_write = ""
    conv_count = 0
    cv1 = ""
    cv3 = ""
    cv3_idx = 0
    for k, v in model.state_dict().items():
        if "num_batches_tracked" not in k and "anchors" not in k and "anchor_grid" not in k:
            vr = v.reshape(-1).cpu().numpy()
            idx = int(k.split(".")[1])
            if ".cv1." in k and ".m." not in k and idx != spp_idx:
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
            if ".cv3." not in k and not (".cv1." in k and ".m." not in k and idx != spp_idx):
                wts_write += "{} {} ".format(k, len(vr))
                for vv in vr:
                    wts_write += " "
                    wts_write += struct.pack(">f", float(vv)).hex()
                wts_write += "\n"
                conv_count += 1
    f.write("{}\n".format(conv_count))
    f.write(wts_write)
