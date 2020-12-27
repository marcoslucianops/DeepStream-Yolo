# Multiple YOLO inferences
How to use multiples GIE's on DeepStream

##

1. Download [my native folder](https://github.com/marcoslucianops/DeepStream-Yolo/tree/master/native), rename to yolo and move to your deepstream/sources folder.
2. Copy each obj.names to deepstream/sources/yolo directory, renaming file to labels_* .txt (* = pgie/sgie1/sgie2/etc), according to each inference type.
3. Copy each yolo.cfg and yolo.weights files to deepstream/sources/yolo directory, renaming files to yolo_* .cfg and yolo_* .weights (* = pgie/sgie1/sgie2/etc), according to each inference type.
4. Make a copy of config_infer_primary.txt file and rename it to config_infer_secondary* .txt (* = 1/2/3/etc), according to inference order.
5. Edit DeepStream for your custom model, according to each yolo_* .cfg (* = pgie/sgie1/sgie2/etc) file: https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/customModels.md

**In example folder, on this repository, have all example files to multiple YOLO inferences.**

##

### Compiling edited models
1. Check your CUDA version (nvcc --version)
2. Go to deepstream/sources/yolo directory.
3. Type command (example for CUDA 10.2 version):

```
CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo
```

##

### Add secondary-gie to deepstream_app_config after primary-gie

Example for 1 secondary-gie (2 inferences):

```
[secondary-gie0]
enable=1
gpu-id=0
gie-unique-id=2
operate-on-gie-id=1
# If you want secodary inference operate on specified class ids of GIE (class ids you want to operate: 1, 1;2, 2;3;4, etc; comment it if you don't want to use)
operate-on-class-ids=0
nvbuf-memory-type=0
config-file=config_infer_secondary1.txt
```

Example for 2 secondary-gie (3 inferences):

```
[secondary-gie0]
enable=1
gpu-id=0
gie-unique-id=2
operate-on-gie-id=1
operate-on-class-ids=0
nvbuf-memory-type=0
config-file=config_infer_secondary1.txt

[secondary-gie1]
enable=1
gpu-id=0
gie-unique-id=3
operate-on-gie-id=1
operate-on-class-ids=0
nvbuf-memory-type=0
config-file=config_infer_secondary2.txt
```

##

### Editing config_infer

* Edit config_infer (config_infer_primary, config_infer_secondary1, etc.) files

Example for primary

```
custom-network-config=yolo_pgie.cfg
model-file=yolo_pgie.weights
model-engine-file=pgie_b16_gpu0_fp16.engine
labelfile-path=labels_pgie.txt
```

Example for secondary1

```
custom-network-config=yolo_sgie1.cfg
model-file=yolo_sgie1.weights
model-engine-file=sgie1_b16_gpu0_fp16.engine
labelfile-path=labels_sgie1.txt
```

Example for secondary2

```
custom-network-config=yolo_sgie2.cfg
model-file=yolo_sgie2.weights
model-engine-file=sgie2_b16_gpu0_fp16.engine
labelfile-path=labels_sgie2.txt
```

##

* Edit gie-unique-id

Example for primary

```
gie-unique-id=1
process-mode=1
```

Example for secondary1

```
gie-unique-id=2
process-mode=2
```

Example for secondary2

```
gie-unique-id=3
process-mode=2
```

##

* Edit batch-size

Example for primary

```
# Number of sources
batch-size=1
```

Example for all secondary:

```
batch-size=16
```

### Testing model
To run your custom YOLO model, use this command

```
deepstream-app -c deepstream_app_config.txt
```

**During test process, engine file will be generated. When engine build process is done, rename engine file according to each configured engine name pgie/sgie1/sgie2/etc) in config_infer file.**
