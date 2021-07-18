# Vrep Object Detection

## Installation 

First off, clone this repo and then  install packages from the requirements using 
```bash 
pip3 install -r requirements.txt
```
Next, go to ObjectDetector folder and run the following command
to clone dependecies from github repo 
```bash 
./config_model.sh
```

Finally, navigate to ObjectDetctor/weights folder and download weights of your choice 

#### SSDMobileNet
```bash 
./donwload_weights.sh -s
```

#### Faster R-CNN
```bash 
./donwload_weights.sh -f
```

#### EfficeintDet V2 
```bash 
./donwload_weights.sh -e
```

## Vrep Config

To obtain images from vrep vision sensor first locate the vrep installtion directory 
and type in the terminal 
```bash 
vrep_installation_dir=<your_installtion_dir>
```
then copy **remoteApi.so** file from your vrep installation directory as follows:
```bash
sudo cp $vrep_installation_dir/programming/remoteApiBindings/lib/lib/Linux/64Bit $vrep_installation_dir/programming/remoteApiBindings/python/python
```



## Tutorial 

see examples folder 

* object_detection_test.py
* vrep_demo_test.py
