# How to use DeepSORT in DeepStream
To use DeepSORT in DeepStream you must first build the engine. You can even choose your own Re-ID Model according to the NVIDIA documentation. 

Unfortunatly since that documentation is licensed we can not adapt or publish any of their READ-ME's without proper license. 

You can find the instructions on how to build this model in the deepstream folder .

```
nano /opt/nvidia/deepstream/deepstream/sources/tracker_DeepSORT/README
```




### Testing model

```
- Enter `samples/configs/deepstream-app/`. In deepstream-app config, change
  [tracker] config to use DeepSORT:
  ll-config-file=config_tracker_DeepSORT.yml
  DeepSORT tracker parameters are in `config_tracker_DeepSORT.yml`.
- Run deepstream-app
  deepstream-app -c <path to config.txt>
```
