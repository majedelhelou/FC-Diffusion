#!/bin/bash

# checkpoint diffusion
(
mkdir -p data/pretrained
cd data/pretrained
gdown https://drive.google.com/uc?id=1norNWWGYP3EZ_o05DmoW1ryKuKMmhlCX # celeba-HQ 256 trained by RePaint
)

# some data and zscores
(
gdown https://drive.google.com/uc?id=1Q_dxuyI41AAmSv9ti3780BwaJQqwvwMv
unzip data.zip
rm data.zip
)