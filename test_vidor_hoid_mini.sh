#!/usr/bin/env bash


export PYTHONPATH=$(pwd):$PYTHONPATH
#echo 'You need to modify <gen_subset.py> and <gpu_id> !!!!!!!!!!!!!!!!!!!!!!!!'
#echo 'You need to modify <gen_subset.py> and <gpu_id> !!!!!!!!!!!!!!!!!!!!!!!!'
#echo 'You need to modify <gen_subset.py> and <gpu_id> !!!!!!!!!!!!!!!!!!!!!!!!'
#echo 'You need to modify <gen_subset.py> and <gpu_id> !!!!!!!!!!!!!!!!!!!!!!!!'
#echo 'You need to modify <gen_subset.py> and <gpu_id> !!!!!!!!!!!!!!!!!!!!!!!!'
#echo 'You need to modify <gen_subset.py> and <gpu_id> !!!!!!!!!!!!!!!!!!!!!!!!'
#echo 'You need to modify <gen_subset.py> and <gpu_id> !!!!!!!!!!!!!!!!!!!!!!!!'
#echo 'You need to modify <gen_subset.py> and <gpu_id> !!!!!!!!!!!!!!!!!!!!!!!!'
#echo 'You need to modify <gen_subset.py> and <gpu_id> !!!!!!!!!!!!!!!!!!!!!!!!'
#echo 'You need to modify <gen_subset.py> and <gpu_id> !!!!!!!!!!!!!!!!!!!!!!!!'
#echo 'You need to modify <gen_subset.py> and <gpu_id> !!!!!!!!!!!!!!!!!!!!!!!!'
#echo 'You need to modify <gen_subset.py> and <gpu_id> !!!!!!!!!!!!!!!!!!!!!!!!'
#echo 'You need to modify <gen_subset.py> and <gpu_id> !!!!!!!!!!!!!!!!!!!!!!!!'


python experiments/fgfa_rfcn/fgfa_rfcn_test.py --cfg experiments/fgfa_rfcn/cfgs/resnet_v1_101_flownet_vidor_hoid_mini_rfcn_end2end_ohem_val.yaml