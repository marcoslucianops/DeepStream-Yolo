"""Reparametrize yolov7 for deployment purposes
credit: https://github.com/WongKinYiu/yolov7/blob/main/tools/reparameterization.ipynb
if you want to reparameterize other yolov7 variants follow the notebook in the above link.

Please place this script under yolov7 directory, since it will use some functions from yolov7 repository.
"""
from copy import deepcopy
from models.yolo import Model
import torch
from utils.torch_utils import select_device, is_parallel
import yaml
import os
import argparse

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="", type=str, help= "path to yolov7 checkpoints file for reparametrization")
    parser.add_argument("--classes", default=80, type=int, help="The number of classes the model was trained on..")
    parser.add_argument("--cfg", default= "cfg/deploy/yolov7.yaml", type=str, help="model configuration file")
    parser.add_argument("--save", default="cfg/deploy/", type=str, help="save reparametrized checkpoints (dir)")
    opt = parser.parse_args()
    return opt

def reparametrize_yolov7(weights, save, cfg, classes):
    """Reparametrize yolov7 model
    yolov7 reparameterization: https://arxiv.org/abs/2207.02696
    
    Args:
        weights: str, path to yolov7 checkpoints file
        classes: int, number of classes you trained your model on..
        cfg: str, path to model configuration file
        save: str, path to directory where you want to save reparametrized checkpoints
        """
    device = select_device('0', batch_size=1)
    ckpt = torch.load(weights, map_location=device)
    model = Model(cfg, ch=3, nc=classes).to(device)

    with open(cfg) as f:
        yml = yaml.load(f, Loader=yaml.SafeLoader)
    anchors = len(yml['anchors'][0]) // 2

    state_dict = ckpt['model'].float().state_dict()
    exclude = []
    intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(intersect_state_dict, strict=False)
    model.names = ckpt['model'].names
    model.nc = ckpt['model'].nc

    # reparametrized YOLOv7 now
    for i in range((model.nc+5)*anchors):
        model.state_dict()['model.105.m.0.weight'].data[i, :, :, :] *= state_dict['model.105.im.0.implicit'].data[:, i, : :].squeeze()
        model.state_dict()['model.105.m.1.weight'].data[i, :, :, :] *= state_dict['model.105.im.1.implicit'].data[:, i, : :].squeeze()
        model.state_dict()['model.105.m.2.weight'].data[i, :, :, :] *= state_dict['model.105.im.2.implicit'].data[:, i, : :].squeeze()
    model.state_dict()['model.105.m.0.bias'].data += state_dict['model.105.m.0.weight'].mul(state_dict['model.105.ia.0.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.1.bias'].data += state_dict['model.105.m.1.weight'].mul(state_dict['model.105.ia.1.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.2.bias'].data += state_dict['model.105.m.2.weight'].mul(state_dict['model.105.ia.2.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.0.bias'].data *= state_dict['model.105.im.0.implicit'].data.squeeze()
    model.state_dict()['model.105.m.1.bias'].data *= state_dict['model.105.im.1.implicit'].data.squeeze()
    model.state_dict()['model.105.m.2.bias'].data *= state_dict['model.105.im.2.implicit'].data.squeeze()

    ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
            'optimizer': None,
            'training_results': None,
            'epoch': -1}

    # save reparameterized checkpoints
    reparametrized_file = os.path.join(save, "yolov7_reparameterized.pt")
    torch.save(ckpt, reparametrized_file)

def main():
    args = read_args()
    print("Reparameterizing yolov7")
    print("--"*40)
    reparametrize_yolov7(args.weights, args.save, args.cfg, args.classes)
    print("done!!")

if __name__ == "__main__":
    main()

