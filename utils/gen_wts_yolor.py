import argparse
import os
import struct
import torch
from utils.torch_utils import select_device
from models.models import Darknet


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch YOLOR conversion (main branch)")
    parser.add_argument("-w", "--weights", required=True, help="Input weights (.pt) file path (required)")
    parser.add_argument("-c", "--cfg", required=True, help="Input cfg (.cfg) file path (required)")
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit("Invalid weights file")
    if not os.path.isfile(args.cfg):
        raise SystemExit("Invalid cfg file")
    return args.weights, args.cfg


pt_file, cfg_file = parse_args()

wts_file = cfg_file.split(".cfg")[0] + ".wts"

device = select_device("cpu")
model = Darknet(cfg_file).to(device)
model.load_state_dict(torch.load(pt_file, map_location=device)["model"])
model.to(device).eval()

with open(wts_file, "w") as f:
    wts_write = ""
    conv_count = 0
    for k, v in model.state_dict().items():
        if not "num_batches_tracked" in k:
            vr = v.reshape(-1).cpu().numpy()
            wts_write += "{} {} ".format(k, len(vr))
            for vv in vr:
                wts_write += " "
                wts_write += struct.pack(">f", float(vv)).hex()
            wts_write += "\n"
            conv_count += 1
    f.write("{}\n".format(conv_count))
    f.write(wts_write)
