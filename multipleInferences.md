# Multiple YOLO inferences
How to use multiples GIE's on DeepStream

##

1. Download [my native folder](https://github.com/marcoslucianops/DeepStream-Yolo/tree/master/native), rename to yolo and move to your deepstream/sources folder.
2. Make a folder, in deepstream/sources/yolo directory, named pgie (where you will put files of primary inference).
3. Make a folder, for each secondary inference, in deepstream/sources/yolo directory, named sgie* (* = 1, 2, 3, etc.; depending on the number of secondary inferences; where you will put files of others inferences).
4. Copy and remane each obj.names file to labels.txt in each inference directory (pgie, sgie*), according each inference type.
5. Copy your yolo.cfg and yolo.weights files to each inference directory (pgie, sgie*), according each inference type.
6. Move nvdsinfer_custom_impl_Yolo folder and config_infer_primary.txt file to each inference directory (pgie, sgie*; for sgie's, rename config_infer_primary to config_infer_secondary*; * = 1, 2, 3, etc.)
7. Edit DeepStream for your custom model, according each yolo.cfg file: https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/customModels.md

**In example folder, on this repository, have all example files to multiple YOLO inferences.**

##

### Editing Makefile
To compile nvdsinfer_custom_impl_Yolo without errors is necessary to edit Makefile (line 34), in nvdsinfer_custom_impl_Yolo folder in each inference directory.
```
CFLAGS+= -I../../includes -I/usr/local/cuda-$(CUDA_VER)/include
```
To:
```
CFLAGS+= -I../../../includes -I/usr/local/cuda-$(CUDA_VER)/include
```

##

### Editing yoloPlugins.h
To run deepstream-app without errors is necessary to edit yoloPlugins.h (line 51), in nvdsinfer_custom_impl_Yolo folder in each secondary inference directory.
```
const char* YOLOLAYER_PLUGIN_VERSION {"1"};
```
To:
```
const char* YOLOLAYER_PLUGIN_VERSION {"2"};
```

Note: 2 = sgie1, 3 = sgie2, 4 = sgie3, etc

##

### Compiling edited models
1. Check your CUDA version (nvcc --version)
2. Go to inference directory.
3. Type command to compile:

* x86 platform
```
CUDA_VER=11.1 make -C nvdsinfer_custom_impl_Yolo
```

* Jetson platform
```
CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo
```

**Do this for each GIE!**

##

### Add secondary-gie to deepstream_app_config after primary-gie

Example for 1 secondary-gie (2 inferences):
```
[secondary-gie0]
enable=1
gpu-id=0
gie-unique-id=2
operate-on-gie-id=1
# If you want secodary inference operate on specified class ids of GIE (class ids you want to operate: 1, 1;2, 2;3;4, 3 etc; comment it if you don't want to use)
operate-on-class-ids=0
nvbuf-memory-type=0
config-file=sgie1/config_infer_secondary1.txt
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
config-file=sgie1/config_infer_secondary1.txt

[secondary-gie1]
enable=1
gpu-id=0
gie-unique-id=3
operate-on-gie-id=1
operate-on-class-ids=0
nvbuf-memory-type=0
config-file=sgie2/config_infer_secondary2.txt
```

Note: remember to edit primary-gie
```
[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
nvbuf-memory-type=0
config-file=config_infer_primary.txt
```

to
```
[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
nvbuf-memory-type=0
config-file=pgie/config_infer_primary.txt
```

##

### Editing config_infer

* Edit path of config (config_infer_primary, config_infer_secondary1, etc.) files

Example for primary

```
custom-network-config=pgie/yolo.cfg
```

Example for secondary1

```
custom-network-config=sgie1/yolo.cfg
```

Example for secondary2

```
custom-network-config=sgie2/yolo.cfg
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

**During test process, engine file will be generated. When engine build process is done, move engine file to respective GIE folder (pgie, sgie1, etc.)**
