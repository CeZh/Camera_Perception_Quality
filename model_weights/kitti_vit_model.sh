#!/bin/bash

% Download the KITTI Dataset Checkpoint Model
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Xh4_r0FLxoamh0EPS8kQl6Kjmlx98rwQ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Xh4_r0FLxoamh0EPS8kQl6Kjmlx98rwQ" -O kitti_vit_model.pt && rm -rf /tmp/cookies.txt
