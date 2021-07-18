#!/usr/bin/env bash

SSDmobileNet='http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz'
FasterRCNN='http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.tar.gz'
EfficientDet='http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz'
download()
{
  # downloading from the web
  wget $1
  # untar
  tar -xf `ls | grep tar`
  # removing tar file
  rm `ls | grep tar`
}

#download $SSDmobileNet

usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }
[ $# -eq 0 ] && usage


detector=""
while getopts ":hsfe" arg; do
  case $arg in
    e) # EfficientDet
      detector="$EfficientDet"
      ;;
    f) # Faster R-CNN
      detector="$FasterRCNN"
      ;;
    s) # SSDMobileNet.
      detector=${SSDmobileNet}
      ;;
    h | *) # Display help.
      usage
      exit 0
      ;;
  esac
done

[[ -z "$detector" ]] && usage && exit 0

echo "[+] downloading weight = $detector"

download $detector

