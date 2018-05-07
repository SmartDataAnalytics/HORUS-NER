#!/bin/bash
source activate horus_27_cv310
pip install spacy && python -m spacy download en
python download_models.py
echo "horus pre-requisites done!"