#!/usr/bin/env bash


export PYTHONPATH=$(pwd):$PYTHONPATH
python experiments/fgfa_rfcn/fgfa_rfcn_test.py --cfg experiments/fgfa_rfcn/cfgs/resnet_v1_101_flownet_imagenet_vid_rfcn_end2end_ohem.yaml
