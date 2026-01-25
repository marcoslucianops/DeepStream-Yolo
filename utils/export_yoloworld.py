import os
import sys
import onnx
import torch
import torch.nn as nn
from copy import deepcopy

from ultralytics import YOLOWorld
from ultralytics.nn.modules import C2f, C2fAttn, WorldDetect, ImagePoolingAttn
import ultralytics.utils
import ultralytics.models.yolo
import ultralytics.utils.tal as _m

sys.modules["ultralytics.yolo"] = ultralytics.models.yolo
sys.modules["ultralytics.yolo.utils"] = ultralytics.utils


def _dist2bbox(distance, anchor_points, xywh=False, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat((x1y1, x2y2), dim)


_m.dist2bbox.__code__ = _dist2bbox.__code__


class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.transpose(1, 2)
        boxes = x[:, :, :4]
        scores, labels = torch.max(x[:, :, 4:], dim=-1, keepdim=True)
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


class YOLOWorldWrapper(nn.Module):
    """Wrapper for YOLOWorld model to handle text embeddings during export."""
    
    def __init__(self, model, txt_feats):
        super().__init__()
        self.model = model
        # Register text features as buffer so they are exported with the model
        self.register_buffer('txt_feats', txt_feats)
    
    def forward(self, x):
        txt_feats = self.txt_feats.to(device=x.device, dtype=x.dtype)
        if txt_feats.shape[0] != x.shape[0]:
            txt_feats = txt_feats.expand(x.shape[0], -1, -1)
        ori_txt_feats = txt_feats.clone()
        
        y = []
        for m in self.model.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)
            y.append(x if m.i in self.model.save else None)
        return x


def yoloworld_export(weights, device, custom_classes=None, fuse=True):
    """Export YOLOWorld model with pre-computed text embeddings.
    
    Args:
        weights: Path to the YOLOWorld .pt file
        device: torch device
        custom_classes: Optional list of custom class names. If None, uses model's default classes.
        fuse: Whether to fuse model layers
    
    Returns:
        wrapped_model: Model ready for ONNX export
        class_names: List of class names used
    """
    print(f"Loading YOLOWorld model: {weights}")
    model = YOLOWorld(weights)
    
    # Set custom classes if provided
    if custom_classes is not None:
        print(f"Setting {len(custom_classes)} custom classes and generating CLIP text embeddings...")
        model.set_classes(custom_classes)
        # Detach text features to ensure no gradient computation during export
        model.model.txt_feats = model.model.txt_feats.detach()
        class_names = custom_classes
        print(f"{'#'*10} Custom Classes {'#'*10}")
        print(f"{class_names}")
        print(f"{'#'*35}\n")
    else:
        # Use model's default classes (COCO 80)
        class_names = list(model.model.names.values()) if isinstance(model.model.names, dict) else model.model.names
        print(f"Using model's default {len(class_names)} classes")
    
    # Get the internal model
    inner_model = deepcopy(model.model).to(device)
    
    for p in inner_model.parameters():
        p.requires_grad = False
    inner_model.eval()
    inner_model.float()
    
    if fuse:
        inner_model = inner_model.fuse()
    
    # Set export mode for all relevant modules
    for k, m in inner_model.named_modules():
        if isinstance(m, WorldDetect):
            m.dynamic = False
            m.export = True
            m.format = "onnx"
        elif isinstance(m, C2f):
            m.forward = m.forward_split
    
    # Get text features (detached to avoid gradient issues)
    txt_feats = inner_model.txt_feats.clone().detach()
    
    # Create wrapper with embedded text features
    wrapped_model = YOLOWorldWrapper(inner_model, txt_feats)
    wrapped_model.eval()
    
    return wrapped_model, class_names


def suppress_warnings():
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)


def main(args):
    suppress_warnings()

    print(f"\nStarting: {args.weights}")

    # Parse custom classes if provided
    custom_classes = None
    if args.custom_classes is not None:
        custom_classes = [item.strip() for item in args.custom_classes.split(',')]

    device = torch.device("cpu")
    model, names = yoloworld_export(args.weights, device, custom_classes)

    if len(names) > 0:
        print("Creating labels.txt file")
        with open("labels.txt", "w", encoding="utf-8") as f:
            for name in names:
                f.write(f"{name}\n")

    model = nn.Sequential(model, DeepStreamOutput())

    img_size = args.size * 2 if len(args.size) == 1 else args.size

    onnx_input_im = torch.zeros(args.batch, 3, *img_size).to(device)
    onnx_output_file = args.weights.rsplit(".", 1)[0] + ".onnx"

    dynamic_axes = {
        "input": {
            0: "batch"
        },
        "output": {
            0: "batch"
        }
    }

    print("Exporting the model to ONNX")
    torch.onnx.export(
        model,
        onnx_input_im,
        onnx_output_file,
        verbose=False,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes if args.dynamic else None
    )

    if args.simplify:
        print("Simplifying the ONNX model")
        import onnxslim
        model_onnx = onnx.load(onnx_output_file)
        model_onnx = onnxslim.slim(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print(f"Done: {onnx_output_file}\n")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="DeepStream YOLOWorld conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with default COCO 80 classes
  python export_yoloworld.py -w yolov8s-worldv2.pt --dynamic

  # Export with custom classes
  python export_yoloworld.py -w yolov8s-worldv2.pt --custom-classes "person, car, dog" --dynamic

  # Export with simplification
  python export_yoloworld.py -w yolov8s-worldv2.pt --custom-classes "person" --dynamic --simplify
        """
    )
    parser.add_argument("-w", "--weights", required=True, type=str, 
                        help="Input weights (.pt) file path (required)")
    parser.add_argument("--custom-classes", type=str, default=None,
                        help='Custom class names, comma-separated (e.g., "person, car, dog")')
    parser.add_argument("-s", "--size", nargs="+", type=int, default=[640], 
                        help="Inference size [H,W] (default [640])")
    parser.add_argument("--opset", type=int, default=17, 
                        help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", 
                        help="ONNX simplify model")
    parser.add_argument("--dynamic", action="store_true", 
                        help="Dynamic batch-size")
    parser.add_argument("--batch", type=int, default=1, 
                        help="Static batch-size")
    args = parser.parse_args()
    
    if not os.path.isfile(args.weights):
        raise RuntimeError("Invalid weights file")
    if args.dynamic and args.batch > 1:
        raise RuntimeError("Cannot set dynamic batch-size and static batch-size at same time")
    
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
