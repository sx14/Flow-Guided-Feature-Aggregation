# --------------------------------------------------------
# Flow-Guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'fgfa_rfcn'))

from fgfa_rfcn import test
import shutil

if __name__ == "__main__":
    from vidvrd_challenge.vidor.gen_subset import prepare_ImageSets

    batch_boundaries = [0, 400, 600, 800, 1000, 1200,
                        1400, 1600, 1800, 2000, 2400]

    batch_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for i in range(1, 10):

        tgt_ds_root = 'data/VidOR'
        tgt_ds_root = os.path.abspath(tgt_ds_root)

        prepare_ImageSets(tgt_ds_root, 'test', batch_boundaries[i], batch_boundaries[i+1])

        cache_path = '../../data/cache'
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)

        print('[%d] test: %d -> %d' % (batch_ids[i], batch_boundaries[i], batch_boundaries[i+1]))
        test.main(batch_id=batch_ids[i])
