#!/usr/bin/env bash

clone_repo()
{
  git clone  https://github.com/tensorflow/models.git --depth 1
}

config_model()
{

  cd models/research
  protoc object_detection/protos/*.proto --python_out=.
  echo "[+] model config successful !"
}
if [ ! -d "models" ] ; then
  echo "[+] cloning repo"
  clone_repo
fi
config_model