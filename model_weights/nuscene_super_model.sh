#!/bin/bash

% Download the NuScene Dataset Checkpoint Model
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZAlrNwRy16S42a-L9dvIw7dxjdt2bfgt' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZAlrNwRy16S42a-L9dvIw7dxjdt2bfgt" -O nuscene_super_model.pt && rm -rf /tmp/cookies.txt

