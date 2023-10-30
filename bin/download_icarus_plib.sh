#!/bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bEKLDN6U-xacqEDDGMdtkHGJ4SrbPW9o' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bEKLDN6U-xacqEDDGMdtkHGJ4SrbPW9o" -O plib.h5
