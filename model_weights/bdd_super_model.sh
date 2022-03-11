#!/bin/bash

% Download the BDD Dataset Checkpoint Model
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1CT3IDlQO1zP-a9Sw-yebpMDvNH0y9LWb' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1CT3IDlQO1zP-a9Sw-yebpMDvNH0y9LWb" -O bdd_super_model.pt && rm -rf /tmp/cookies.txt

