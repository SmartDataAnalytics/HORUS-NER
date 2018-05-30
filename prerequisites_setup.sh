#!/bin/bash
source activate horus_27_cv310
pip install spacy && python -m spacy download en
python download_models.py
conda install -c conda-forge dlib=19.10
conda install pytorch-cpu torchvision-cpu -c pytorch
echo "horus pre-requisites done!"