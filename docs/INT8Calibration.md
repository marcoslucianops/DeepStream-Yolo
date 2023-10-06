# INT8 calibration (PTQ)

### 1. Install OpenCV

```
sudo apt-get install libopencv-dev
```

### 2. Compile/recompile the `nvdsinfer_custom_impl_Yolo` lib with OpenCV support

* DeepStream 6.3 on x86 platform

  ```
  CUDA_VER=12.1 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.2 on x86 platform

  ```
  CUDA_VER=11.8 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.1.1 on x86 platform

  ```
  CUDA_VER=11.7 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.1 on x86 platform

  ```
  CUDA_VER=11.6 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.0.1 / 6.0 on x86 platform

  ```
  CUDA_VER=11.4 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 5.1 on x86 platform

  ```
  CUDA_VER=11.1 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.3 / 6.2 / 6.1.1 / 6.1 on Jetson platform

  ```
  CUDA_VER=11.4 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.0.1 / 6.0 / 5.1 on Jetson platform

  ```
  CUDA_VER=10.2 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
  ```

### 3. For COCO dataset, download the [val2017](https://drive.google.com/file/d/1gbvfn7mcsGDRZ_luJwtITL-ru2kK99aK/view?usp=sharing), extract, and move to DeepStream-Yolo folder

* Select 1000 random images from COCO dataset to run calibration

  ```
  mkdir calibration
  ```

  ```
  for jpg in $(ls -1 val2017/*.jpg | sort -R | head -1000); do \
      cp ${jpg} calibration/; \
  done
  ```

* Create the `calibration.txt` file with all selected images

  ```
  realpath calibration/*jpg > calibration.txt
  ```

* Set environment variables

  ```
  export INT8_CALIB_IMG_PATH=calibration.txt
  export INT8_CALIB_BATCH_SIZE=1
  ```

* Edit the `config_infer` file

  ```
  ...
  model-engine-file=model_b1_gpu0_fp32.engine
  #int8-calib-file=calib.table
  ...
  network-mode=0
  ...
  ```

    To

  ```
  ...
  model-engine-file=model_b1_gpu0_int8.engine
  int8-calib-file=calib.table
  ...
  network-mode=1
  ...
  ```

* Run

  ```
  deepstream-app -c deepstream_app_config.txt
  ```

**NOTE**: NVIDIA recommends at least 500 images to get a good accuracy. On this example, I recommend to use 1000 images to get better accuracy (more images = more accuracy). Higher `INT8_CALIB_BATCH_SIZE` values will result in more accuracy and faster calibration speed. Set it according to you GPU memory. This process may take a long time.
