# YOLOv4-large usage

**NOTE**: Follow this guide for yolov4-tiny from [PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)

- [Convert model](#convert-model)
- [Compile the lib](#compile-the-lib)
- [Edit the config_infer_primary file](#edit-the-config_infer_primary-file)
- [Edit the deepstream_app_config file](#edit-the-deepstream_app_config-file)
- [Testing the model](#testing-the-model)
- [Implementation details about [route] layers in yolov4-tiny](#implementation-details-about-route-layers-in-yolov4-tiny)


##

### Convert model

#### 1. Download the PyTorch_YOLOv4 repo and install the requirements

```
git clone https://github.com/WongKinYiu/PyTorch_YOLOv4
cd PyTorch_YOLOv4
pip3 install -r requirements.txt
```

**NOTE**: It is recommended to use Python virtualenv.


#### 2. Copy conversor

Copy the `gen_wts_yoloV4-tiny.py` file from `DeepStream-Yolo/utils` directory to the `PyTorch_YOLOv4` folder.

#### 3. Download the model

Fetch some `pt` weights for PyTorch_YOLOv4. There are none in the official repo but I found some [here]
(https://github.com/tjuskyzhang/Scaled-YOLOv4-TensorRT/tree/master/yolov4-tiny-tensorrt). Nonetheless, note this process has been tested with both these weights and a model trained from scratch on the official PyTorch_YOLOv4 repo.

Because PyTorch_YOLOv4's `.pt` format does not carry the model architecture, we need an additional `.cfg` file. This file is probably available in the PyTorch_YOLOv4/cfg repo.

**NOTE**: You can use your custom model, but it is important to keep the YOLO model reference (`yolov4_`) in you `cfg` and `weights`/`wts` filenames to generate the engine correctly.
**NOTE**: Commonly available YOLOv4-tiny cfg files have a typo making the second Yolo layer use anchors 1,2,3 rather than 0,1,2. Change it if you wish.

#### 4. Convert model

Generate the `cfg` and `wts` files from the `.pt` file. Here's how to run the script:

```
# Parameters explained
python3 gen_wts_yoloV4-tiny.py -w weights.pt     # REQUIRED: input weights
                               -s width [height] # OPTIONAL: image size, you can specify just the width for square imgs
                               -n 80             # OPTIONAL: number of classes
                               -c config.cfg     # OPTIONAL: path to input cfg file

# Valid examples
python3 gen_wts_yoloV4-tiny.py -w yolov4-tiny.pt
python3 gen_wts_yoloV4-tiny.py -w yolov4-tiny.pt -s 480
python3 gen_wts_yoloV4-tiny.py -w yolov4-tiny.pt -s 480 480 -n 80 -c custom-yolov4-tiny.cfg
```

The only required argument is the input weight files, as the rest can be automatically fetched:

- Image size: can be inferred from the model training history packed in the `.pt`.
- Number of classes: can be inferred from the number of weights of some layers in the `.pt`.
- Input `cfg` file: if not passed, a file with the same name as the weights minus extensions will be fetched from `cfg/`

The generated `cfg` and `wts` carry the `nvds` prefix to avoid confusion between input and output cfg files.

**NOTE**: If your model fails to build an engine, try passing all arguments by hand.

#### 5. Copy generated files

Copy the generated `cfg` and `wts` files to the `DeepStream-Yolo` folder.

##

### Compile the lib

Open the `DeepStream-Yolo` folder and compile the lib

* DeepStream 6.1 on x86 platform

  ```
  CUDA_VER=11.6 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.0.1 / 6.0 on x86 platform

  ```
  CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.1 on Jetson platform

  ```
  CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.0.1 / 6.0 on Jetson platform

  ```
  CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo
  ```

##

### Edit the config_infer_primary file

Edit the `config_infer_primary.txt` file according to your model:

```
[property]
...
custom-network-config=nvds-yolov4-tiny.cfg
model-file=nvds-yolov4-tiny.wts
...
```

##

### Edit the deepstream_app_config file

Make sure this file points to the one modified above.

```
...
[primary-gie]
...
config-file=config_infer_primary.txt
```

##

### Testing the model

```
deepstream-app -c deepstream_app_config.txt
```


## Implementation details about [route] layers in yolov4-tiny
*This section is not needed to run Deepstream with YOLOv4-tiny*

YOLOv4 tiny has two widely used implementations: one in Darknet and other in Pytorch. These are similar but have a different special `route` layer, which accomplish the same goal but are implemented differently. This layer is called `[route_lhalf]`, and takes *half* the feature maps from the preceding layer and passes it to the next one. In the cfg files, they are implemented as:

**Torch**: official torch implementation [yolov4-tiny.cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4-tiny.cfg)
```
[route_lhalf]
layer=-1      # Take half of the layer -1
```

**Darknet**: standard `[route]` modified [yolov4-tiny.cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny.cfg)
```
[route]
layer=-1    # Take the layer -1
groups=2    # Divide its channels in 2 groups
group_id=1  # Pass the 2nd to the next layer (0-indexed)
```

These implement the same concept, only Torch's implementation takes the 'left' half of feature maps, while Darknet's takes the 'right' one. This makes Darknet and Torch weights incompatible ([source1](https://github.com/tjuskyzhang/Scaled-YOLOv4-TensorRT/issues/5#issuecomment-677975656), [source2](https://github.com/WongKinYiu/ScaledYOLOv4/issues/165)).

The script used in this tutorial uses `[route_lhalf]` layers, which come by default in PyTorch_YOLOv4 `cfg/` architectures, and are implemented in this repo at `nvdsinfer_custom_impl_Yolo/layers/route_lhalf_layer.cpp`. Alternatively, one might use a `cfg` with standard `[route]` layers implemented at `nvdsinfer_custom_impl_Yolo/layers/route_layer.cpp`, setting `group_id=0`.
