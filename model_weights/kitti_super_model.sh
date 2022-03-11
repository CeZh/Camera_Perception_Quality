#!/bin/bash

% Download the KITTI Dataset Checkpoint Model
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1k0WeCL5qgDLxlvhayfrJC7BlfW57TXb5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1k0WeCL5qgDLxlvhayfrJC7BlfW57TXb5" -O kitti_super_model.pt && rm -rf /tmp/cookies.txt

